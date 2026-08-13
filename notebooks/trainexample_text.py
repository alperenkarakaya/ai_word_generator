"""
Device: cuda
Vocab size: 8000
Model params: 13.81M
/content/uni_bitirme/transformer/train.py:111: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=device == "cuda")
/content/uni_bitirme/transformer/train.py:129: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(enabled=use_amp, dtype=torch.float16):
step      0 | loss 9.0494 | lr 1.20e-06 | tok/s 7404
step     50 | loss 8.1727 | lr 3.12e-05 | tok/s 93060
step    100 | loss 7.3354 | lr 6.12e-05 | tok/s 105791
step    150 | loss 6.4889 | lr 9.12e-05 | tok/s 110609
step    200 | loss 6.0969 | lr 1.21e-04 | tok/s 113012
step    250 | loss 5.9444 | lr 1.51e-04 | tok/s 114222
step    300 | loss 5.6654 | lr 1.81e-04 | tok/s 114864
step    350 | loss 5.7089 | lr 2.11e-04 | tok/s 115309
step    400 | loss 5.4771 | lr 2.41e-04 | tok/s 115440
step    450 | loss 5.4439 | lr 2.71e-04 | tok/s 115469
/content/uni_bitirme/transformer/train.py:49: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(enabled=use_amp, dtype=torch.float16):
  >>> step 500 | train 5.3269 | val 5.3895
  >>> en iyi val. Kaydedildi: transformer.pt
step    500 | loss 5.2663 | lr 3.00e-04 | tok/s 107605
step    550 | loss 5.3366 | lr 2.99e-04 | tok/s 107947
step    600 | loss 5.2956 | lr 2.97e-04 | tok/s 108129
step    650 | loss 5.2627 | lr 2.93e-04 | tok/s 108162
step    700 | loss 5.1018 | lr 2.88e-04 | tok/s 108229
step    750 | loss 5.2232 | lr 2.82e-04 | tok/s 108171
step    800 | loss 5.0047 | lr 2.74e-04 | tok/s 108011
step    850 | loss 5.0178 | lr 2.65e-04 | tok/s 107760
step    900 | loss 4.9617 | lr 2.55e-04 | tok/s 107505
step    950 | loss 4.9493 | lr 2.44e-04 | tok/s 107215
  >>> step 1000 | train 4.8245 | val 4.9470
  >>> en iyi val. Kaydedildi: transformer.pt
step   1000 | loss 4.8697 | lr 2.32e-04 | tok/s 102857
step   1050 | loss 4.8754 | lr 2.20e-04 | tok/s 102616
step   1100 | loss 4.8464 | lr 2.06e-04 | tok/s 102294
step   1150 | loss 4.9038 | lr 1.93e-04 | tok/s 101916
step   1200 | loss 4.8016 | lr 1.79e-04 | tok/s 101481
step   1250 | loss 4.7600 | lr 1.65e-04 | tok/s 100979
step   1300 | loss 4.6751 | lr 1.51e-04 | tok/s 100420
step   1350 | loss 4.7475 | lr 1.37e-04 | tok/s 99804
step   1400 | loss 4.6515 | lr 1.23e-04 | tok/s 99176
step   1450 | loss 4.7070 | lr 1.10e-04 | tok/s 98549
  >>> step 1500 | train 4.5808 | val 4.7123
  >>> en iyi val. Kaydedildi: transformer.pt
step   1500 | loss 4.5527 | lr 9.73e-05 | tok/s 95504
step   1550 | loss 4.7073 | lr 8.54e-05 | tok/s 95230
step   1600 | loss 4.6741 | lr 7.45e-05 | tok/s 95037
step   1650 | loss 4.6797 | lr 6.45e-05 | tok/s 94893
step   1700 | loss 4.5678 | lr 5.56e-05 | tok/s 94797
step   1750 | loss 4.6126 | lr 4.79e-05 | tok/s 94734
step   1800 | loss 4.6083 | lr 4.16e-05 | tok/s 94691
step   1850 | loss 4.5165 | lr 3.65e-05 | tok/s 94655
step   1900 | loss 4.5708 | lr 3.29e-05 | tok/s 94627
step   1950 | loss 4.6352 | lr 3.07e-05 | tok/s 94599
  >>> step 2000 | train 4.4740 | val 4.6395
  >>> en iyi val. Kaydedildi: transformer.pt
Eğitim tamamlandı.
"""
