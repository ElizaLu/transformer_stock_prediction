from pathlib import Path
import pickle

from master_model import MASTER
from trainer import SequenceTrainer


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    universe = "csi300"
    prefix = "opensource"
    seed = 0

    dl_test = load_pickle(Path("data") / prefix / f"{universe}_dl_test.pkl")

    d_feat = 158
    d_model = 256
    t_nhead = 4
    s_nhead = 2
    dropout = 0.5
    gate_input_start_index = 158
    gate_input_end_index = 221
    beta = 5 if universe == "csi300" else 2

    trainer = SequenceTrainer(
        model=MASTER(
            d_feat=d_feat,
            d_model=d_model,
            t_nhead=t_nhead,
            s_nhead=s_nhead,
            T_dropout_rate=dropout,
            S_dropout_rate=dropout,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            beta=beta,
        ),
        n_epochs=1,
        lr=1e-5,
        GPU=0,
        seed=seed,
        train_stop_loss_thred=0.95,
    )

    param_path = Path("model") / f"{universe}_{seed}.pkl"
    trainer.load_param(param_path)
    preds, metrics = trainer.predict(dl_test)
    print(metrics)


if __name__ == "__main__":
    main()