WANDB_API_KEY=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 poetry run python plonk/train.py \
    exp=osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann \
    dataset.train_dataset.root=/home/debian/data/geoloc/osv5m/wds/streetclip/train \
    dataset.test_dataset.root=/home/debian/data/geoloc/osv5m/wds/streetclip/test \
    dataset.val_dataset.root=/home/debian/data/geoloc/osv5m/wds/streetclip/test \
    dataset.embedding_name=streetclip \
    mode=traineval