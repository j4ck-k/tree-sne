import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from gensim.models.keyedvectors import KeyedVectors


### Getting MNIST Data

def get_mnist():
    mnist = fetch_openml('mnist_784')
    X = mnist.data
    X /= 255
    y = mnist.target.astype('int')

    U, s, V = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return np.dot(U, np.diag(s))[:,:50], y
    

### Getting HathiTrust Data - Download full files from https://zenodo.org/records/1477018

def get_full_meta(path):

    return pd.read_csv(path, sep='\t', on_bad_lines='warn')

def get_literature_meta(meta):

    return meta[meta['lc1'].isin(['PR', 'PS'])][:100000]

def get_shakespeare_meta(meta):
    
    authors = [
        'Kyd, Thomas, 1558-1594.',
        'Marlowe, Christopher, 1564-1593.',
        'Jonson, Ben, 1573?-1637.',
        'Dekker, Thomas, approximately 1572-1632.',
        'Webster, John, 1580?-1625?',
        'Beaumont, Francis, 1584-1616.',
        'Fletcher, John, 1579-1625.',
        'Massinger, Philip, 1583-1640.',
        'Ford, John, 1586-ca. 1640.',
        'Bacon, Francis, 1561-1626.',
        'Bacon, Francis, 1561-1626',
        'Donne, John, 1572-1631.',
        'Spenser, Edmund, 1552?-1599.',
        'Sidney, Philip, Sir, 1554-1586.',
        'Lyly, John, 1554?-1606.',
        'Middleton, Thomas, d. 1627.',
        'Nash, Thomas, 1567-1601.',
        'Peele, George, 1556-1596.',
        'Raleigh, Walter, Sir, 1552?-1618.',
        'Herbert, George, 1593-1633.',
        'Chapman, George, 1559?-1634.',
        'Heywood, Thomas, approximately 1574-1641.',
        'Herrick, Robert, 1591-1674.',
        'Butler, Samuel, 1612-1680.',
        'Daniel, Samuel, 1562-1619.',
        'Burton, Robert, 1577-1640.',
        'Wither, George, 1588-1667.',
        'Shirley, James, 1596-1666.',
        'Shakespeare, William, 1564-1616'
        ]

    return meta[(meta['language'] == 'English') & (meta['lc1'] == 'PR') & (meta['first_author_name'].isin(authors))]

def get_model(path, meta):
    
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    X = np.zeros((meta['id'].shape[0], 100))
    for i,ident in enumerate(meta['id']):
        X[i,:] = model[ident]
    
    return X


if __name__ == '__main__':

    meta = get_full_meta('hathi-data/hathi.tsv')
    lit_meta = get_literature_meta(meta)
    sh_meta = get_shakespeare_meta(meta)

    X_lit = get_model('hathi-data/hathi_pca.bin', lit_meta)
    X_sh = get_model('hathi-data/hathi_pca.bin', sh_meta)

    lit_meta.to_pickle('hathi-data/lit-meta.pkl')
    sh_meta.to_pickle('hathi-data/shakespeare-meta.pkl')

    np.save('hathi-data/lit-model.npy', X_lit)
    np.save('hathi-data/shakespeare-model.npy', X_sh)