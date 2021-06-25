import torch
old_model = torch.load("ensemble/latest.pth")

n = dict()
for k, v in old_model.items():
    # if not k == 'state_dict':
    #     n[k] = v
    # else:
    #     n['state_dict'] = dict()
    #     for k1, v1 in v.items():
    #         n['state_dict']['module.fusion.' + k] = v
    if k == 'state_dict':
        for k1, v1 in v.items():
            n['module.' + k1] = v1

torch.save(n, "ensemble/latest_fixed.pth")
