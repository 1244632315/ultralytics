# XT Tool Scripts

- `tools_compare_xt_ab.py`: 对比不同 XT 方案（ablation）。
- `tools_prepare_xt_norm1.py`: 生成 `xt_norm1` 数据。
- `tools_prepare_xt_sc.py`: 生成 `x_sc`（`[X, sin(pi*tau), cos(pi*tau)]`）数据。
- `tools_run_t_only_and_report.py`: 训练 `t_only` 并导出对比报告。
- `tools_run_xt_fix1.py`: 旧版 XT 对比/修复实验脚本。
- `tools_run_xt_sc_full_and_compare.py`: `x/t/xt/x_sc` 全量训练与对比。
- `tools_run_angle_loss_v1.py`: `angle_loss` v1 全量训练与基线对比。
- `tools_build_xt_seq_msod.py`: 基于 XT 仿真生成 `x-only` 的 `MSOD` 风格时序分割数据集。
- `tools_visualize_xt_seq_msod.py`: 抽样导出时序图像与 `mask` 叠加预览。
- `tools_run_compare_msod_x.py`: 启动 `CSAUNet/DNANet/DnTNet/MSAMNet`，或做无 GPU 的 dataloader smoke test。
