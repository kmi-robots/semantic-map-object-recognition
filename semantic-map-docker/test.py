import torch
from embedding_extractor import query_embedding
import os


def test(model, model_checkpoint, path_to_test, device, trans, path_to_train_embeds, K=5):

    # Code for test/inference time on one(few) shot(s)
    # for each query image
    train_embeds = torch.load(path_to_train_embeds)
    class_wise_res = []


    with open('pt_results/ranking_log.txt', 'w') as outr:

        for root, dirs, files in os.walk(path_to_test):

            if files:  #For each object class

                #tot_wrong = 0

                classname = str(root.split('/')[-1])

                outr.write(classname)

                for file in files: #For each example in that class

                    qembedding = query_embedding(model, model_checkpoint, os.path.join(root, file), \
                                                 device, transforms=trans)
                    # Similarity matching against indexed data

                    similarities = {}

                    for emb_id, emb in train_embeds.items():

                        # Cos sim reduces to dot product since embeddings are L2 normalized
                        similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

                    # Return top-K results
                    ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

                    outr.write("The %i most similar objects to the provided image are: \n" % K)

                    correct_preds = 0

                    class_accs = []

                    if K == 1:

                        key, val = ranking[0]
                        label = key.split("_")[0]  # [:-1]
                        outr.write(label + ": " + str(val) + "\n")

                        if label == classname:

                            correct_preds += 1

                        else:

                            outr.write('{} mistaken for {}'.format(classname, label))
                            # tot_wrong += 1

                        class_accs.append(correct_preds)

                    else:
                        for key, val in ranking[:K - 1]:


                            label = key.split("_")[0] #[:-1]

                            outr.write(label + ": " + str(val) + "\n")

                            #Parse labels to binary as correct? Yes/No
                            if label == classname:

                                correct_preds += 1

                            else:

                                outr.write('{} mistaken for {}'.format(classname, label))
                                #tot_wrong += 1

                            avg_acc = correct_preds/K
                            class_accs.append(avg_acc)

                macro_avg = sum(class_accs)/len(class_accs)

                print('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
                outr.write('Mean average accuracy for class {} is {}'.format(classname, float(macro_avg)))
                class_wise_res.append((classname, macro_avg))

    return zip(*class_wise_res)