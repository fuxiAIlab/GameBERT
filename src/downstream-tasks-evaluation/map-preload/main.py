from trainer import ModelTrainer
import logging
import argparse
from hyperparams import Hyperparams
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Consumer")
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--model_tag', type=str, required=True)
    args = parser.parse_args()

    server = int(args.server)
    model_tag = args.model_tag

    train_size = {
        242: 411199,
        204: 467903
    }
    test_size = {
        242: 372573,
        204: 512064
    }

    hp = Hyperparams(server, model_tag, train_size=train_size[server], test_size=test_size[server])

    logger = logging.getLogger(__name__)
    logging.root.setLevel(level=logging.INFO)

    model_trainer = ModelTrainer(hp)
    logger.info('model trainer initialized.')

    model_trainer.model_train()
    logger.info('model trainer done.')
