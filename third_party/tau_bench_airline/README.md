# τ-bench airline 原版工具（只读参考）

本目录是 [τ-bench](https://github.com/sierra-research/tau-bench)（Sierra，MIT 协议，提交 `59a200c`）
`tau_bench/envs/airline/tools/` 的**原样拷贝**，版权声明见 `LICENSE`。**不要修改这里的文件。**

用途：`agent/tests/test_airline_tools.py` 用同样的输入分别调用这里的原版工具和本项目的 `MySQLBackend`，
比对两边的结果是否一致。加载方式见 `agent/tests/taubench_ref.py`（不需要安装整个 tau-bench 包）。
