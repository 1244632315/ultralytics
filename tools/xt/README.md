# XT Tool Scripts

- `TRAJECTORY_EVAL_PROTOCOL.md`: 轨迹级对比实验协议，统一检测法与分割法的主评价口径。
- `tools_eval_trajectory_protocol.py`: 基于现有 `bbox` 轨迹摘要结果，输出统一的轨迹级主表（`TDR/CTR/TC/EHR/Frag/FTPS`）。
- `tools_eval_kbs_detector_bbox.py`: KBS 上的 detector 轨迹框评估；现支持 `--pred-track-mode reconstruct`，从 detector 结果重建逐帧支持并导出 `pred_tracks`。
- `tools_build_kbs_expanded_eval_json.py`: 生成 KBS 扩展测试集 json，默认用于“官方 train/test 合并后，排除 finetune 用过的序列”。

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
