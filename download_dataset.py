from huggingface_hub import snapshot_download

# 下载整个数据集到指定文件夹
path = snapshot_download(
    repo_id="lxxiao/272-dim-HumanML3D",
    repo_type="dataset",
    local_dir="./humanml3d_272"
)

print(f"数据集已下载至: {path}")