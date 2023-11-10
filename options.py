import argparse
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument("--root-dir", help="Image dir")
parser.add_argument("--split-type", help="Type of data split", default="A")
parser.add_argument("--label-file", help="Label file path")
parser.add_argument("--img-channel", help="Image channels", type=int, default=3)
parser.add_argument("--img-width", help="Image width", type=int, default=768)
parser.add_argument("--img-height", help="Image height", type=int, default=224)

# Training
parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-4)
parser.add_argument("--num-epochs", help="Number of training epochs", type=int, default=10)
parser.add_argument("--decay-rate", help="Decay rate", type=float, default=0.9)
parser.add_argument("--batch-size", help="Batch size", type=int, default=32)
parser.add_argument("--log-every", help="Log the training process every n steps", type=int, default=10)
parser.add_argument("--val-every", help="Validation step every n steps", type=int, default=100)
parser.add_argument("--save-every", help="Save model weights every n epochs", type=int, default=1)
parser.add_argument("--out-dir", help="Output directory training process", default="outputs")
parser.add_argument("--max-length", help="Maximum length of predicted sequence", type=int, default=100)
parser.add_argument("--lr-step-every", help="Step lr every n steps", type=int, default=100)

# Model
parser.add_argument("--stn-on", help="Using TPS as transformation or not", action="store_true")
parser.add_argument("--weights", help="Load weights into model", default='')

# Template
parser.add_argument("--task", help="Task name", required=True)