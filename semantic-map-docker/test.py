import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import average_precision_score
from embedding_extractor import query_embedding


def test(model, model_checkpoint, test_loader, path_to_query_data, device, trans, path_to_train_embeds, K=5):

    # Code for test/inference time on one(few) shot(s)
    # for each query image
    qembedding = query_embedding(model, model_checkpoint, path_to_query_data, \
                                 device, transforms=trans)

    train_embeds = torch.load(path_to_train_embeds)

    # Similarity matching against indexed data

    similarities = {}

    for emb_id, emb in train_embeds.items():
        # Cos sim reduces to dot product since embeddings are L2 normalized
        similarities[emb_id] = torch.mm(qembedding, emb.t()).item()

    # Return top-K results
    ranking = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    print("The %i most similar objects to the provided image are: \n" % K)

    for key, val in ranking[:K - 1]:
        label = key.split("_")[0][:-1]
        print(label + ": " + str(val) + "\n")

