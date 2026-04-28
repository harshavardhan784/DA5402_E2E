"""
src/register_clip_model.py
───────────────────────────
Register the CLIP + probe model in the MLflow Model Registry.

Usage:
    python src/register_clip_model.py \
        --run_id <mlflow_run_id> \
        --mode linear_probe \
        --faiss_index_img data/faiss/index_img.bin \
        --faiss_index_txt data/faiss/index_txt.bin \
        --faiss_meta  data/faiss/meta.json \
        --model_name  ViT-B-32 \
        --pretrained  openai

After registration, serve with:
    mlflow models serve -m "models:/clip_product_retrieval/Production" -p 5001
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from clip_mlflow_wrapper import CLIPRetrieverModel

src_dir = Path(__file__).parent

def register(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    # Download the checkpoint from the training run
    if args.mode == "linear_probe":
        ckpt_local = mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="checkpoints/probe_best.pt")
    elif args.mode == "finetune":
        ckpt_local = mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="checkpoints/finetune_best.pt")
    else:
        ckpt_local = "zero_shot"   # sentinel value

    # Determine corpus text-embedding path (optional, logged during update_faiss_index)
    # Anchor to the project root (two levels up from src/register_clip_model.py)
    project_root   = src_dir.parent
    corpus_emb_dir = project_root / "data" / "corpus_embeddings"
    txt_emb_path   = corpus_emb_dir / "corpus_text_embeddings.npy"


    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "model_config.json"
        cfg_path.write_text(json.dumps({
            "model_name":    args.model_name,
            "pretrained":    args.pretrained,
            "mode":          args.mode,
            "embed_dim":     args.embed_dim,
            "probe_hidden":  args.probe_hidden,
            "probe_dropout": args.probe_dropout,
        }))

        artifacts = {
            "clip_checkpoint":  ckpt_local,
            "faiss_index_img":  args.faiss_index_img,
            "faiss_index_txt":  args.faiss_index_txt,
            "faiss_meta":       args.faiss_meta,
            "model_config":     str(cfg_path),
        }

        # Include text embeddings if available (used for text-similarity ranking)
        if txt_emb_path.exists():
            artifacts["corpus_text_embeddings"] = str(txt_emb_path)

        pip_reqs = [
            "open-clip-torch",
            "faiss-cpu",
            "torch",
            "Pillow",
            "numpy",
            "pandas",
        ]

        with mlflow.start_run(run_name=f"register_{args.mode}"):
            mlflow.pyfunc.log_model(
                artifact_path         = "clip_retriever",
                python_model          = CLIPRetrieverModel(),
                artifacts             = artifacts,
                pip_requirements      = pip_reqs,
                registered_model_name = "clip_product_retrieval",
                code_paths            = [
                    str(src_dir / "clip_experiments.py"),
                    str(src_dir / "clip_mlflow_wrapper.py"),
                ],
            )


    # Transition to Production
    versions = client.search_model_versions("name='clip_product_retrieval'")
    latest   = sorted(versions, key=lambda v: int(v.version))[-1]
    # Replace transition_model_version_stage with set_registered_model_alias
    client.set_registered_model_alias(
        name    = "clip_product_retrieval",
        alias   = "Production",
        version = latest.version,
    )

    print(f"Registered model v{latest.version} → Production")
    print(f"Serve with:  mlflow models serve -m 'models:/clip_product_retrieval/Production' -p 5001")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_id",        required=True)
    p.add_argument("--mode",          default="linear_probe",
                   choices=["zero_shot", "linear_probe", "finetune"])
    p.add_argument("--faiss_index_img",   default="data/faiss/index_img.bin")
    p.add_argument("--faiss_index_txt",   default="data/faiss/index_txt.bin")
    p.add_argument("--faiss_meta",    default="data/faiss/meta.json")
    p.add_argument("--model_name",    default="ViT-B-32")
    p.add_argument("--pretrained",    default="openai")
    p.add_argument("--embed_dim",     type=int,   default=512)
    p.add_argument("--probe_hidden",  type=int,   default=None)
    p.add_argument("--probe_dropout", type=float, default=0.1)
    p.add_argument("--tracking_uri",  default="http://mlflow:5000")
    register(p.parse_args())