"""Train a detector on prepared feature pairs, not on raw review clips."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader
from src.models.detector import AudioVisualDetector
from src.training.dataset import PairedFeatureDataset, collate_pairs
from src.training.loop import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True, help='New experiment directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--consistency-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.lr) <= 0 or args.consistency_weight < 0:
        parser.error('Positive epochs/batch/hidden/lr and non-negative consistency weight required')
    torch.manual_seed(args.seed)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    train = PairedFeatureDataset(args.manifest, 'train')
    val = PairedFeatureDataset(args.manifest, 'val')
    # Preflight every cache to catch mixed checkpoints and dimensions across batches.
    signature = None
    for dataset in (train, val):
        for item in dataset:
            current = (item['extractor_id'], item['audio'].shape[1], item['visual'].shape[1])
            if signature is not None and signature != current:
                raise ValueError('All caches must share extractor and feature dimensions')
            signature = current
    if {row['label'] for row in train.rows} != {'0', '1'}:
        raise ValueError('Training requires both real and fake samples')
    config = dict(audio_dim=signature[1], visual_dim=signature[2], hidden_dim=args.hidden_dim)
    model = AudioVisualDetector(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_pairs)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_pairs)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    best = float('inf')
    with (output / 'epochs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'val_supervised'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            trained = run_epoch(model, train_loader, optimizer, args.consistency_weight)
            validated = run_epoch(model, val_loader)
            row = dict(epoch=epoch, train_loss=trained['loss'], val_supervised=validated['supervised'])
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
            if validated['supervised'] < best:
                best = validated['supervised']
                torch.save(dict(model=model.state_dict(), model_config=config, epoch=epoch,
                                args=vars(args), extractor_id=signature[0], val_supervised=best,
                                manifest_sha256=manifest_hash),
                           output / 'best.pt')


if __name__ == '__main__':
    main()
