import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Inference Options')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--video', type=str, help='Path to the input video')
    parser.add_argument('--camera', action='store_true', help='Enable camera detection')
    parser.add_argument('--algorithm', choices=['nn', 'rf', 'knn','gbm'],
                        default='nn', help='Select the classification algorithm (default: nn)')

    return parser.parse_args()


