import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd


def assign(X, centers, batch_size=2048):
    labels = np.empty(len(X), dtype=int)
    distances = np.empty(len(X))
    center_norms = np.einsum('ij,ij->i', centers, centers)
    for start in range(0,len(X),batch_size):
        batch = X[start:start+batch_size]
        squared = np.einsum('ij,ij->i',batch,batch)[:,None] + center_norms - 2*batch@centers.T
        np.maximum(squared,0,out=squared)
        selected = squared.argmin(axis=1)
        labels[start:start+len(batch)] = selected
        distances[start:start+len(batch)] = squared[np.arange(len(batch)),selected]
    return labels,distances


def initialize(X,k,rng):
    """k-Means++: select subsequent seeds proportional to squared distance."""
    centers = np.empty((k,X.shape[1]))
    centers[0] = X[rng.integers(len(X))]
    nearest = np.einsum('ij,ij->i',X-centers[0],X-centers[0])
    for i in range(1,k):
        total = nearest.sum()
        index = rng.choice(len(X),p=nearest/total) if total > 0 else rng.integers(len(X))
        centers[i] = X[index]
        squared = np.einsum('ij,ij->i',X-centers[i],X-centers[i])
        nearest = np.minimum(nearest,squared)
    return centers


class KMeans:
    def __init__(self,n_clusters=20,n_init=5,max_iter=200,tol=1e-5,random_state=42):
        for name,value in [('n_clusters',n_clusters),('n_init',n_init),('max_iter',max_iter)]:
            if not isinstance(value,int) or isinstance(value,bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        self.n_clusters,self.n_init,self.max_iter=n_clusters,n_init,max_iter
        self.tol,self.random_state=tol,random_state

    def fit(self,X,verbose=False):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]==0 or not np.isfinite(X).all():
            raise ValueError('X must be a finite 2D matrix')
        if len(np.unique(X,axis=0)) < self.n_clusters:
            raise ValueError('Need at least n_clusters distinct vectors')
        rng=np.random.default_rng(self.random_state)
        self.inertia_=float('inf')
        self.runs_=[]
        for restart in range(self.n_init):
            if verbose: print(f'Initialization {restart+1}/{self.n_init}...',flush=True)
            centers=initialize(X,self.n_clusters,rng)
            previous=None
            for iteration in range(1,self.max_iter+1):
                labels,squared=assign(X,centers)
                if previous is not None and np.array_equal(labels,previous):
                    break
                updated=np.empty_like(centers)
                empty=[]
                for c in range(self.n_clusters):
                    rows=X[labels==c]
                    if len(rows): updated[c]=rows.mean(axis=0)
                    else: empty.append(c)
                # Re-seed empty clusters at distinct, poorly represented rows.
                if empty:
                    farthest=np.argsort(-squared,kind='stable')
                    for c,row in zip(empty,farthest): updated[c]=X[row]
                shift=np.linalg.norm(updated-centers,axis=1).max()
                centers=updated
                previous=labels.copy()
                if shift <= self.tol: break
            labels,squared=assign(X,centers)
            inertia=float(squared.sum())
            self.runs_.append(dict(initialization=restart+1,iterations=iteration,inertia=inertia))
            if verbose: print(f'  iterations={iteration}, inertia={inertia:.6f}',flush=True)
            if inertia < self.inertia_:
                self.inertia_=inertia
                self.cluster_centers_=centers.copy()
                self.labels_=labels.copy()
                self.n_iter_=iteration
                self.best_initialization_=restart+1
        return self

    def predict(self,X):
        if not hasattr(self,'cluster_centers_'): raise ValueError('Call fit first')
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=self.cluster_centers_.shape[1] or not np.isfinite(X).all():
            raise ValueError('Prediction features must match training features')
        return assign(X,self.cluster_centers_)[0]


def pair_count(counts):
    counts=np.asarray(counts,dtype=np.float64)
    return float((counts*(counts-1)/2).sum())


def agreement(table):
    """Contingency table rows=clusters, columns=true labels."""
    table=np.asarray(table,dtype=float)
    n=table.sum()
    row,col=table.sum(axis=1),table.sum(axis=0)
    together=pair_count(table)
    same_label=pair_count(col)
    same_cluster=pair_count(row)
    expected=same_label*same_cluster/(n*(n-1)/2) if n>1 else 0
    maximum=(same_label+same_cluster)/2
    ari=(together-expected)/(maximum-expected) if maximum>expected else 1.
    a,b=np.nonzero(table)
    mi=float(np.sum(table[a,b]/n*np.log(table[a,b]*n/(row[a]*col[b]))))
    def entropy(counts):
        p=counts[counts>0]/n
        return float(-np.sum(p*np.log(p)))
    hrow,hcol=entropy(row),entropy(col)
    return dict(adjusted_rand_index=ari,
                normalized_mutual_information=2*mi/(hrow+hcol) if hrow+hcol else 1.,
                cluster_purity=float(table.max(axis=1).sum()/n),
                same_label_pair_coassignment=together/same_label if same_label else 1.,
                same_cluster_pair_same_label=together/same_cluster if same_cluster else 1.,
                mean_label_dominant_cluster_fraction=float((table.max(axis=0)/col).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('--clusters',type=int,default=20)
    parser.add_argument('--n-init',type=int,default=5)
    parser.add_argument('--max-iter',type=int,default=200)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',type=Path,default=Path('kmeans_results'))
    args=parser.parse_args()
    frame=pd.read_csv(args.data)
    features=[c for c in frame if c.startswith('feature_')]
    if not features or 'id' not in frame or 'label' not in frame:
        parser.error('CSV needs id, feature_* columns, and label for AFTER-FITTING evaluation')
    X=frame[features].to_numpy(dtype=np.float64)
    if frame.label.isna().any() or not frame.id.is_unique:
        parser.error('Need nonmissing labels and unique IDs')
    started=perf_counter()
    model=KMeans(args.clusters,args.n_init,args.max_iter,random_state=args.seed).fit(X,verbose=True)
    elapsed=perf_counter()-started
    # This is the first point where known labels enter the experiment.
    labels=frame.label.to_numpy()
    classes=np.unique(labels)
    encoded=np.searchsorted(classes,labels)
    counts=np.zeros((args.clusters,len(classes)),dtype=int)
    np.add.at(counts,(model.labels_,encoded),1)
    table=pd.DataFrame(counts,index=pd.Index(range(args.clusters),name='cluster'),
                       columns=[f'label_{value}' for value in classes])
    cluster_rows=[]
    for c,row in enumerate(counts):
        size=int(row.sum())
        ranked=np.argsort(-row,kind='stable')
        top=[dict(label=int(classes[i]),count=int(row[i]),percent=round(100*row[i]/size,2))
             for i in ranked[:5] if row[i]>0] if size else []
        cluster_rows.append(dict(cluster=c,size=size,dominant_label=top[0]['label'] if top else None,
                                 dominant_count=top[0]['count'] if top else 0,
                                 purity_percent=top[0]['percent'] if top else 0,
                                 labels_present=int((row>0).sum()),top_five_labels=json.dumps(top)))
    label_rows=[]
    for j,label in enumerate(classes):
        row=counts[:,j]
        c=int(row.argmax())
        label_rows.append(dict(label=int(label),total=int(row.sum()),dominant_cluster=c,
                               count_in_dominant_cluster=int(row[c]),
                               percent_in_dominant_cluster=100*row[c]/row.sum(),
                               clusters_present=int((row>0).sum())))
    # Chance comparison preserves actual cluster sizes and true label counts.
    permutation_rng=np.random.default_rng(args.seed+1)
    permuted=[]
    for _ in range(100):
        null=np.zeros_like(counts)
        np.add.at(null,(model.labels_,permutation_rng.permutation(encoded)),1)
        permuted.append(agreement(null))
    metrics=agreement(counts)
    metrics.update(n_rows=len(X),n_features=X.shape[1],n_classes=len(classes),n_clusters=args.clusters,
                   seed=args.seed,n_init=args.n_init,max_iter=args.max_iter,
                   preprocessing='none; use supplied features as provided',
                   min_vector_norm=float(np.linalg.norm(X,axis=1).min()),
                   max_vector_norm=float(np.linalg.norm(X,axis=1).max()),
                   fit_seconds=elapsed,inertia=model.inertia_,best_initialization=model.best_initialization_,
                   best_iterations=model.n_iter_,runs=model.runs_,
                   chance_means={k:float(np.mean([r[k] for r in permuted])) for k in agreement(counts)},
                   chance_permutations=100,
                   cluster_purity_upper_bound=float(np.sort(counts.sum(axis=0))[-args.clusters:].sum()/len(X)))
    dest=args.output_dir
    dest.mkdir(parents=True,exist_ok=True)
    table.to_csv(dest/'cluster_label_counts.csv')
    pd.DataFrame(cluster_rows).to_csv(dest/'cluster_summary.csv',index=False)
    pd.DataFrame(label_rows).to_csv(dest/'label_summary.csv',index=False)
    pd.DataFrame({'id':frame.id,'label':labels,'cluster':model.labels_}).to_csv(dest/'image_cluster_assignments.csv',index=False)
    np.save(dest/'cluster_centers.npy',model.cluster_centers_)
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
    print('\nClusters:')
    print(pd.DataFrame(cluster_rows).drop(columns='top_five_labels').to_string(index=False))
    print('\nLabels most concentrated in one cluster:')
    print(pd.DataFrame(label_rows).sort_values('percent_in_dominant_cluster',ascending=False).head(10).to_string(index=False))


if __name__=='__main__': main()
