# PyTorch template for quick starting

This template provides basic structure of a PyTorch project including checkpoint saving and logging.

Use the following command line to activate `run.py` for training / testing the deep learning network. Detail explaination can be check by `--help`.

```shell
python run.py [-te | --training] [-cp | --checkpoing_path <path>] [-re | --resume] [-ep | --epoch <integer>] [-bs | --batch_size <integer>] [-lr | --learning_rate <float>] [-wd | --weight_decay <float>] [-rp | --result_path <path-result>]
```

Beside basic deep learning training hyperparameters (learning rate, epoch, etc.), you can add parser arguments in `utils/arguments.py` according to your requirement.

**Notice:** Configuations settings will be saved at checkpoint path once the training procedure is started. In order to preserve essential configuations, you can edit `utils/arguments/Configs->self.REQUIRE_CONFGS` for the save procedure.

This template has been applied on MNIST dataset for digits recognition and it ran smoothly. Feel free to use it!

## Training

The simplest way to start training is:

```shell
python run.py 
```

You can use `-cp` or `--checkpoing_path` to set the checkpoint saving path, and use `-re` or `--resume` to resume the training procedure from checkpoint saving path. You can also set hyperparameters with `-ep` or `--epoch`, `-bs` or `--batch_size`, etc.

## Testing

Use:

```shell
python run.py [-te | --testing]
```

to start testing. You can appoint checkpoing file for evaluation using `-cp` or `--checkpoing_path`, and use `-rp` or `--result_path` to set path for saving evaluation result.