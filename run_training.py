import torch
from torch.optim import Adam
from model import Model, compute_loss
from random import shuffle, seed
from util import get_names
from clearml import Task
from run_game import Evaluator
from torch.cuda.amp import autocast, GradScaler
from loader import BatchSeqLoader
import os

MULTIPLIER = 40
ENVS = 32
STEPS = 32
BATCH = 32
OPT_STEPS = 1
LEARNING_RATE = 0.001
STEPS_PER_EPOCH = 128
LOAD_PATH = "stress_model/Epoch510ModelData.tm"

PROFILE = True
WRITE = False
LOAD = True

data_dir = "replays/training_replays"
#reference_replay = "replays/training_replays/ffeb0ec0d691a3c5d741cc069d176dae44132f73f02332a39fbc15657865e92f.SC2Replay/stats.pkl"
reference_replay = "stats_replay"


def selective_load(source_dict, target):
    '''
    load_dict = {}
    target_dict = target.state_dict()
    for x in target_dict:
        print(x)
        if x in source_dict and source_dict[x].shape == target_dict[x].shape:
            load_dict[x] = source_dict[x]
    '''
    target.load_state_dict(source_dict)


def run_training(data_dir, reference_replay):
    torch.manual_seed(2138)
    seed(2137)
    print("Initializing")

    names = get_names(data_dir)
    names = list(sorted(names))
    evaluator = Evaluator(reference_replay=reference_replay)

    model = Model()
    print("Loaded Model")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    start_epoch = 0
    print("Loaded Optimizier")
    if LOAD:
        checkpoint = torch.load(LOAD_PATH)
        selective_load(checkpoint['model_state_dict'], model)
        selective_load(checkpoint['optimizer_state_dict'], optimizer)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        observations_count = checkpoint['observations_count']
        print("Loaded saved Model, Epoch: ", checkpoint['epoch'])
    else:
        observations_count = 0

    loader = BatchSeqLoader(ENVS, names, STEPS, BATCH, model)
    
    if WRITE:
        task = Task.init(project_name='StarStress',
                            task_name='train supervised')
        logger = task.get_logger()
    # 32 - 1.46 in 100 epochs
    CUDA = True

    if CUDA:
        print("Training with GPU")
        model.cuda()

    total_epoch_loss = 0
    total_losses_dict = {}
    total_scores_dict = {}
    steps = 0

    optimizer.zero_grad()
    scaler = GradScaler()
    print("Starting Training")
    for epoch in range(start_epoch, 1000000, 1):
        running_loss = 0
        # grad_magnitudes = 0
        # needs to be divisble by 4 - shouldn't change much, but might improve
        # optimizer performance
        for i in range(STEPS_PER_EPOCH):
            inputs, targets, masks, hiddens, tot_result = loader.get_batch()
            steps += STEPS * BATCH

            with autocast():
                output, new_hiddens = model(inputs, hiddens, tot_result, targets)
                #output, new_hiddens = model(inputs, hiddens, targets)
                #print(len(inputs))
                #print(type(inputs))
                #print(output)
                loss, losses_dict, scores_dict = compute_loss(
                    output, targets, masks)
            reduced_loss = loss / OPT_STEPS / BATCH
            scaler.scale(reduced_loss).backward()


            loss_val = loss.item()
            running_loss += loss_val
            total_epoch_loss += loss_val

            del inputs
            del targets
            del masks

            if len(total_losses_dict) > 0:
                for x in total_losses_dict:
                    total_losses_dict[x] += losses_dict[x].item()
                    total_scores_dict[x] += scores_dict[x].item()
            else:
                total_losses_dict = {
                    x: losses_dict[x].item()
                    for x in losses_dict
                }
                total_scores_dict = {
                    x: scores_dict[x].item()
                    for x in scores_dict
                }
            if i % OPT_STEPS == OPT_STEPS - 1:  # print every 4 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / OPT_STEPS / BATCH))

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                running_loss = 0.0

            loader.put_back(new_hiddens)

        weights = 0
        for param in model.parameters():
            weights += (param * param).sum().item()

        logger.report_scalar(title="Stats",
                             series="Weight Magnitude",
                             value=weights,
                             iteration=epoch)
        # writer.add_scalar("Stats/Gradient Magnitude",
        #                   grad_magnitudes/steps,epoch)
        print(f"avg loss in epoch {total_epoch_loss/steps}", "Total observations:", observations_count)
        logger.report_scalar(title="Losses",
                             series="Total",
                             value=total_epoch_loss / steps,
                             iteration=epoch)
        for x in total_losses_dict:
            logger.report_scalar(title="Losses",
                                 series=x,
                                 value=total_losses_dict[x] /
                                 (total_scores_dict[x] + 0.001),
                                 iteration=epoch)

        eval_res = evaluator.try_get_results()
        if eval_res is not None:
            d, ep = eval_res
            for key in d:
                logger.report_scalar(title="Wins",
                                     series=key,
                                     value=d[key],
                                     iteration=ep)

        #if evaluator.best_score > 0 or epoch % 120 == 0:
        #    if not evaluator.running:
        #        print("STARTING EVALUATOR")
        #        evaluator.start_eval(model, epoch)

        total_losses_dict = {}
        total_scores_dict = {}
        observations_count += steps
        PATH = "stress_model/ModelData.tm"
        torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'observations_count': observations_count,
                }, PATH)
        if (epoch) % 10 == 0:
            PATH = "stress_model/" + "Epoch" + str(epoch) + "ModelData.tm"

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'observations_count': observations_count,
                }, PATH)
        print("SAVED")
        steps = 0
        total_epoch_loss = 0


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("source_dir", help="A directory from which features will be loaded for training")
    #parser.add_argument("reference_replay", help="A single processed replay that will be used in evaluation")
    #args = parser.parse_args()
    
    os.makedirs("stress_model", exist_ok=True)
    run_training(data_dir, reference_replay)

    #run_training(args.source_dir, args.reference_replay)