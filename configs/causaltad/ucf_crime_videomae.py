_base_ = [
    "../_base_/datasets/ucf_crime/features_videomae_test.py",  # dataset config
    "../_base_/models/causaltad4ucfcrime.py",  # model config
]

model = dict(
    projection=dict(in_channels=1408, input_pdrop=0.3),
    rpn_head=dict(loss_normalizer=250),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
)

optimizer = dict(type="AdamW", lr=3e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=7, max_epoch=75)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=40,
)

work_dir = "exps/ucf/causal_videomae"
