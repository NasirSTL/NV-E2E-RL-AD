import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path", default='./train_val_txt')
    parser.add_argument("--model_type", help="Model type", default='ENet')
    parser.add_argument("--loss_type", help="Loss type", default='FocalLoss')
    parser.add_argument("--save", required=False, help="Directory to save model", default="./log")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=200)
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--bs", required=False, type=int, help="Batch size", default=32)
    parser.add_argument("--val", required=False, type=bool, help="Use validation", default=False)
    parser.add_argument("--lr", required=False, type=float, help="Learning rate", default=0.00005)
    parser.add_argument("--pretrained", required=False, default=None, help="pretrained model path")
    parser.add_argument("--img", default="./test_output/input.jpg", help="test image path")
    parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")
    return parser.parse_args()
