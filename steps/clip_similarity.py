"""计算相邻帧的 CLIP 余弦相似度"""
import os
import csv


def compute_clip_cosine(image_path1, image_path2, model_name="ViT-B/32", device=None, model=None, preprocess=None):
    import torch
    from PIL import Image
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        raise FileNotFoundError("image path not found")
    if model is None or preprocess is None:
        try:
            import clip as clip_oa
            model, preprocess = clip_oa.load(model_name, device=device)
        except ModuleNotFoundError:
            import open_clip
            ocl_model = model_name.replace("/", "-")
            model, _, preprocess = open_clip.create_model_and_transforms(ocl_model, pretrained="laion2b_s34b_b79k")
            model = model.to(device)

    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    t1 = preprocess(img1).unsqueeze(0).to(device)
    t2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        f1 = model.encode_image(t1)
        f2 = model.encode_image(t2)

    f1 = F.normalize(f1, dim=-1)
    f2 = F.normalize(f2, dim=-1)
    sim = F.cosine_similarity(f1, f2).item()
    return sim


def compute_dir_adjacent_similarities(dir_path, output_csv, model_name="ViT-B/32", device=None, limit=None):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import clip as clip_oa
        model, preprocess = clip_oa.load(model_name, device=device)
    except ModuleNotFoundError:
        import open_clip
        ocl_model = model_name.replace("/", "-")
        model, _, preprocess = open_clip.create_model_and_transforms(ocl_model, pretrained="laion2b_s34b_b79k")
        model = model.to(device)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            files.append(name)
    files.sort()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image1", "image2", "similarity"])
        count = 0
        n = len(files)
        max_pairs = n - 1
        if limit is not None:
            max_pairs = min(max_pairs, int(limit))
        for i in range(max_pairs):
            p1 = os.path.join(dir_path, files[i])
            p2 = os.path.join(dir_path, files[i + 1])
            sim = compute_clip_cosine(p1, p2, model_name=model_name, device=device, model=model, preprocess=preprocess)
            w.writerow([files[i], files[i + 1], f"{sim:.6f}"])
            count += 1
    return count
