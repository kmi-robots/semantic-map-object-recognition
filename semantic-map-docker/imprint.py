import torch


def imprint(model, device, data_loader):

    model.eval()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(data_loader):

            target = torch.max(target, 1)[1]
            input_anchor = data[0].to(device)

            l2norm_emb = model.extract(input_anchor)

            if batch_idx == 0:
                output_stack = l2norm_emb
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, l2norm_emb), 0)

            target_stack = torch.cat((target_stack, target), 0)

    #Supported for 10 classes here
    new_weight = torch.zeros(10, 256)
    for i in range(10):
        tmp = output_stack[target_stack == i].mean(0) #Take the average example if more than one is present
        new_weight[i] = tmp / tmp.norm(p=2) #L2 normalize again

    #Use created template/weight matrix to initialize last classification layer
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))
    model.fc2.weight.data = weight