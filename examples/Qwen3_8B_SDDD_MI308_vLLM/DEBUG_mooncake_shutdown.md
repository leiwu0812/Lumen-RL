# Debug 记录: 训练结束后 mooncake RPC_FAIL 刷屏

日志: `output/Qwen3_8B_SDDD/LumenRL/qwen3-8b-eagle3-vllm-mi308.log`
现象: `LumenRL finished.` 之后 25+ 分钟一直刷:
```
E master_client.cpp:279] Client not available
... MasterClient::FetchTasks response: status=failed, error=RPC_FAIL, latency≈4500000us
... MasterClient::Ping     response: status=failed, error=RPC_FAIL, latency≈4500000us
E client_service.cpp:2924] Failed to ping master for N times (non-HA); reconnecting to 127.0.1.1:51339
E client_service.cpp:2929] Reconnect failed to 127.0.1.1:51339: RPC_FAIL
```
直到 N 累到 155 还在涨。

---

## 1. 关键 timeline

```
06:38:19  step=4 finished; SpecDistillTrainer.train finished after 5 steps
06:38:21  trainer 端 mooncake client (id=740042059177303195) UnmountSegment 干净退出
          FilereadWorkerPool / MemcpyWorkerPool destroyed   ← 干净
06:38:22  VllmTeacherEngine: shutdown complete   ← Python wrapper 自认为退出了
06:38:22  E master_client.cpp:284] RPC call failed: End of file   ← 第一条 error
06:38:22  SpecDistillTrainer.cleanup complete
06:38:22  LumenRL finished.
06:38:22 → 07:03+ (持续 25 min)  vLLM 侧 mooncake client 不停尝试 ping / reconnect ...
```

每条 RPC 卡 4.5 s (mooncake 默认 RPC 超时)，ping 失败 10 次会触发一次 reconnect，循环刷。

## 2. 定位是谁在刷

整个 run 里有两个 mooncake client，可以从 `client_id` 区分:

| client_id 前缀 | 持有进程 | 角色 | 退出情况 |
|---|---|---|---|
| `740042059177303195-…` | 主进程 pid 81 (trainer 侧 `EagleMooncakeStore`，kv consumer) | 读 teacher hidden states | **干净 UnmountSegment 退出** |
| `9243099054434557898-…` | vLLM Worker pid 1192 (`MooncakeHiddenStatesConnector`，`kv_role=kv_producer`) | 写 teacher hidden states | **没退出，背景线程一直 ping/fetch** |

刷屏的全部来自后者。它有两个常驻 C++ daemon 线程:
- pid 2119 → `MasterClient::Ping` (每秒)
- pid 2121 → `MasterClient::FetchTasks` (每秒)

## 3. 根因

清理顺序 + 进程组处理都有问题:

1. 训练侧 `EagleMooncakeStore` 调了 `UnmountSegment` + 释放 worker pool → 干净。
2. `lumenrl/transfer/mooncake_master.py` 先被关掉（其实它的 `os.killpg`+SIGTERM 是对的）→ master 服务端没了。
3. 此时 vLLM teacher 子进程里那份 mooncake producer client 还活着；它的 Ping 线程下一次 RPC 立刻收到 `RPC call failed: End of file`，随后进入无限重试。
4. **关键 bug**:`VllmTeacherEngine.shutdown()` 里只对 wrapper 子进程发了 SIGTERM
   ```python
   self._proc.terminate()
   self._proc.wait(timeout=10)
   ```
   但 `subprocess.Popen(..., start_new_session=True)` 把 wrapper 放在了自己的 session/进程组。vLLM 在 wrapper 内部通过 `multiproc_executor` (spawn) 又拉起 `EngineCore` (pid 930) 和 `Worker_TP0..3` (pid 1192-1195)。SIGTERM 只到 wrapper 一个进程，wrapper 立即死亡，**孙子辈被 reparent 到 init**:
   - EngineCore 可能正常退出；
   - `Worker_TP0` 里的 mooncake producer client 没人通知它 `disconnect()`，C++ 线程是非 daemon 的，永远活着。
5. `VllmTeacherEngine: shutdown complete` 这行只代表 Python wrapper 自认为退出了，**和 vLLM worker 实际状态没关系**。
6. `torchrun` / 外层 `tee` 因此也不返回，log 持续追加。

简言之: **master 先死 → producer client 进入永久重连循环 + 父进程没有用 killpg 杀进程组 → 孤儿 worker + 孤儿 C++ 线程**。

## 4. 影响

- 训练完整跑完、checkpoint 已存、`LumenRL finished` 正常打了 → **训练本身 OK**。
- 唯一副作用: 脚本不返回 + log 越涨越大。需要手动 `kill` 掉残留的 vLLM worker / EngineCore 进程。

## 5. 修复

两个文件:

### `lumenrl/engine/inference/vllm_teacher_engine.py`
- 加 `import signal`
- `shutdown()` 重写:
  - graceful shutdown 命令改成 **fire-and-forget** (直接 write+flush，不再走 `_send_cmd` 阻塞读 resp) —— worker 卡死时不再吊死调用方
  - 等 3 s 让 worker 自己退；不退则 `os.killpg(getpgid(proc.pid), SIGTERM)` 杀整个进程组，覆盖 EngineCore + Worker_TP0..3 + 它们的 mooncake C++ 线程
  - 再等 5 s 不退则 SIGKILL；再 5 s 不退打 warning (不再静默泄漏)
  - mooncake_store / FIFO 文件关闭逻辑保留
- 新增 `_killpg()` 静态方法集中处理 `ProcessLookupError` / `PermissionError`
- `__del__` 也改用 `_killpg(..., SIGKILL)`，与上面一致

```python
def shutdown(self) -> None:
    proc = self._proc
    if proc is not None and proc.poll() is None:
        cmd_f = self._cmd_f
        if cmd_f is not None:
            try:
                cmd_f.write(json.dumps({"cmd": "shutdown"}) + "\n")
                cmd_f.flush()
            except Exception:
                pass

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass

        if proc.poll() is None:
            self._killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "vLLM teacher subprocess pid=%d did not exit "
                        "after SIGKILL; orphan workers may remain.",
                        proc.pid,
                    )
    # ... mooncake_store.close() + FIFO 清理保持不变 ...

@staticmethod
def _killpg(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass

def __del__(self) -> None:
    proc = getattr(self, "_proc", None)
    if proc is not None and proc.poll() is None:
        self._killpg(proc.pid, signal.SIGKILL)
```

### `lumenrl/trainer/spec_distill_trainer.py`
- `cleanup()` 给 teacher 和 master 各包了 try/except + 注释解释为什么必须先 teacher 后 master，避免一边失败导致另一边漏关
- 顺序不变 (teacher → master)，这本来就是对的

```python
def cleanup(self) -> None:
    """Teacher engine must shut down BEFORE the mooncake master so the
    vLLM-side Mooncake producer client disconnects while its master
    is still up; otherwise its background Ping/FetchTasks threads
    enter an infinite RPC_FAIL retry loop and prevent process exit.
    """
    if self._teacher_engine is not None:
        try:
            self._teacher_engine.shutdown()
        except Exception as e:
            logger.warning("[rank %d] teacher_engine.shutdown failed: %s",
                           self._rank, e)
        self._teacher_engine = None
    if self._mooncake_master is not None:
        try:
            self._mooncake_master.shutdown()
        except Exception as e:
            logger.warning("[rank %d] mooncake_master.shutdown failed: %s",
                           self._rank, e)
        self._mooncake_master = None
    # ... 其余清理保持不变 ...
```

## 6. 验证方法

下次 run 完后应当满足:
- log 在 `LumenRL finished.` 之后不再追加 mooncake RPC_FAIL 行；
- `run_qwen3_8b.sh` 自然返回，`tee` 退出；
- `ps -ef | grep -E "vllm|mooncake_master"` 没有残留进程。

如仍有残留，可临时兜底:
```bash
# 在 run_qwen3_8b.sh 末尾加
trap 'pkill -P $$; pkill -9 -f "vllm|mooncake_master" || true' EXIT
```
