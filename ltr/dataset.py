from dataclasses import dataclass
import numpy as np
import gc
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter, namedtuple
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer


def load_data_in_libsvm_format(
    data_path=None, file_prefix=None, feature_size=-1, topk=100
):
    features = []
    dids = []
    initial_list = []
    qids = []
    labels = []
    initial_scores = []
    initial_list_lengths = []
    feature_fin = open(data_path)
    qid_to_idx = {}
    line_num = -1
    for line in feature_fin:
        line_num += 1
        arr = line.strip().split(" ")
        qid = arr[1].split(":")[1]
        if qid not in qid_to_idx:
            qid_to_idx[qid] = len(qid_to_idx)
            qids.append(qid)
            initial_list.append([])
            labels.append([])

        # create query-document information
        qidx = qid_to_idx[qid]
        if len(initial_list[qidx]) == topk:
            continue
        initial_list[qidx].append(line_num)
        label = int(arr[0])
        labels[qidx].append(label)
        did = qid + "_" + str(line_num)
        dids.append(did)

        # read query-document feature vectors
        auto_feature_size = feature_size == -1

        if auto_feature_size:
            feature_size = 5

        features.append([0.0 for _ in range(feature_size)])
        for x in arr[2:]:
            arr2 = x.split(":")
            feature_idx = int(arr2[0]) - 1
            if feature_idx >= feature_size and auto_feature_size:
                features[-1] += [0.0 for _ in range(feature_idx - feature_size + 1)]
                feature_size = feature_idx + 1
            if feature_idx < feature_size:
                features[-1][int(feature_idx)] = float(arr2[1])

    feature_fin.close()

    initial_list_lengths = [len(initial_list[i]) for i in range(len(initial_list))]

    ds = {}
    ds["fm"] = np.array(features)
    ds["lv"] = np.concatenate([np.array(x) for x in labels], axis=0)
    ds["dlr"] = np.cumsum([0] + initial_list_lengths)
    return ds


class Preprocess:
    def __init__(
        self, sw_path, tokenizer=WordPunctTokenizer(), stemmer=PorterStemmer()
    ) -> None:
        with open(sw_path, "r") as stw_file:
            stw_lines = stw_file.readlines()
            stop_words = set([l.strip().lower() for l in stw_lines])
        self.sw = stop_words
        self.tokenizer = tokenizer
        self.stemmer = stemmer

    def pipeline(
        self, text, stem=True, remove_stopwords=True, lowercase_text=True
    ) -> list:
        tokens = []
        for token in self.tokenizer.tokenize(text):
            if remove_stopwords and token.lower() in self.sw:
                continue
            if stem:
                token = self.stemmer.stem(token)
            if lowercase_text:
                token = token.lower()
            tokens.append(token)

        return tokens


# ToDo: Complete the implemenation of process_douments method in the Documents class
class Documents:
    def __init__(self, preprocesser: Preprocess) -> None:
        self.preprocesser: Preprocess = preprocesser
        self.index = defaultdict(defaultdict)  # Index
        self.dl = defaultdict(int)  # Document Length
        self.df = defaultdict(int)  # Document Frequencies
        self.num_docs = 0  # Number of all documents

    def process_documents(self, doc_path: str):
        """Preprocess the document collection.
        Preprocess the collection file (document information). Calculates and updates
        all of the class attributes in the __init__ function:
        - index: terms for each document
        - dl: document lengths
        - df: document frequencies (how many documents contain each term)
        - num_docs: total number of documents
        
        Parameters
        ----------
        doc_path : str
            Path of the file holding documents ID and their corresponding text
        """
        with open(doc_path, "r") as doc_file:
            for line in tqdm(doc_file, desc="Processing documents"):
                
                # BEGIN SOLUTION
                split = line.strip().split("\t")
                doc_id, doc_text = split[0], split[1]
                
                # Preprocess the document text (tokenize, stem, remove stopwords)
                doc_terms = self.preprocesser.pipeline(doc_text)
                
                # Store the preprocessed terms in the index
                self.index[doc_id] = doc_terms
                
                # Update document length
                self.dl[doc_id] = len(doc_terms)
                
                # Increment document count
                self.num_docs += 1
                
                # Update document frequencies for each unique term in this document
                # df stores how many documents contain each term
                for term in set(doc_terms):
                    self.df[term] = self.df[term] + 1
                # END SOLUTION


class Queries:
    def __init__(self, preprocessor: Preprocess) -> None:
        self.preprocessor = preprocessor
        self.qmap = defaultdict(list)
        self.num_queries = 0

    def preprocess_queries(self, query_path):
        with open(query_path, "r") as query_file:
            for line in query_file:
                qid, q_text = line.strip().split("\t")
                q_text = self.preprocessor.pipeline(q_text)
                self.qmap[qid] = q_text
                self.num_queries += 1


__feature_list__ = [
    "bm25",
    "query_term_coverage",
    "query_term_coverage_ratio",
    "stream_length",
    "idf",
    "sum_stream_length_normalized_tf",
    "min_stream_length_normalized_tf",
    "max_stream_length_normalized_tf",
    "mean_stream_length_normalized_tf",
    "var_stream_length_normalized_tf",
    "sum_tfidf",
    "min_tfidf",
    "max_tfidf",
    "mean_tfidf",
    "var_tfidf",
]


@dataclass
class FeatureList:
    f1 = "bm25"  # BM25 value. Parameters: k1 = 1.5, b = 0.75
    f2 = "query_term_coverage"  # number of query terms in the document
    f3 = "query_term_coverage_ratio"  # Ratio of # query terms in the document to # query terms in the query.
    f4 = "stream_length"  # length of document
    f5 = "idf"  # sum of document frequencies
    f6 = "sum_stream_length_normalized_tf"  # Sum over the ratios of each term to document length
    f7 = "min_stream_length_normalized_tf"
    f8 = "max_stream_length_normalized_tf"
    f9 = "mean_stream_length_normalized_tf"
    f10 = "var_stream_length_normalized_tf"
    f11 = "sum_tfidf"  # Sum of tfidf
    f12 = "min_tfidf"
    f13 = "max_tfidf"
    f14 = "mean_tfidf"
    f15 = "var_tfidf"


class FeatureExtraction:
    def __init__(self, features: dict, documents: Documents, queries: Queries) -> None:
        self.features = features
        self.documents = documents
        self.queries = queries
        self.avg_doc_len = sum(self.documents.dl.values()) / len(self.documents.dl) if len(self.documents.dl) > 0 else 0

    # TODO Implement this function
    def extract(self, qid: str, docid: str, **args) -> dict:
        """Extract features for a query-document pair.
        For each query and document, extract the features requested and store them
        in self.features attribute.

        Parameters
        ----------
        qid : str
            Query ID
        docid : str
            Document ID

        Returns
        -------
        dict
            Dictionary of features
        """

        # BEGIN SOLUTION
        # Initialize features for this query-doc pair if not already present
        if qid not in self.features:
            self.features[qid] = {}
        if docid not in self.features[qid]:
            self.features[qid][docid] = {}
        
        # Extract all features
        for feature in __feature_list__:
            match feature:
                case FeatureList.f1:
                    self.compute_bm25(qid, docid, **args)
                case FeatureList.f2:
                    self.compute_query_term_coverage(qid, docid, **args)
                case FeatureList.f3:
                    self.compute_query_term_coverage_ratio(qid, docid, **args)
                case FeatureList.f4:
                    self.compute_stream_length(qid, docid, **args)
                case FeatureList.f5:
                    self.compute_idf(qid, docid, **args)
                case FeatureList.f6:
                    self.compute_sum_stream_length_normalized_tf(qid, docid, **args)
                case FeatureList.f7:
                    self.compute_min_stream_length_normalized_tf(qid, docid, **args)
                case FeatureList.f8:
                    self.compute_max_stream_length_normalized_tf(qid, docid, **args)
                case FeatureList.f9:
                    self.compute_mean_stream_length_normalized_tf(qid, docid, **args)
                case FeatureList.f10:
                    self.compute_var_stream_length_normalized_tf(qid, docid, **args)
                case FeatureList.f11:
                    self.compute_sum_tfidf(qid, docid, **args)
                case FeatureList.f12:
                    self.compute_min_tfidf(qid, docid, **args)
                case FeatureList.f13:
                    self.compute_max_tfidf(qid, docid, **args)
                case FeatureList.f14:
                    self.compute_mean_tfidf(qid, docid, **args)
                case FeatureList.f15:
                    self.compute_var_tfidf(qid, docid, **args)
                case _:
                    print(f"Feature {feature} not found")
                    pass
        
        return self.features[qid][docid]
    
    def compute_bm25(self, qid, docid, k1=1.5, b=0.75, **args):
        """Compute BM25 score for a query-document pair.
        BM25 formula: sum_t( IDF(t) * ((k1 + 1) * tf) / (k1 * (1 - b + b * (dl/avgdl)) + tf) )
        where t is each term in the query, tf is term frequency in the document,
        dl is document length, and avgdl is average document length.
        """
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = self.documents.dl[docid]
        
        # Skip if document not found
        if not doc_terms:
            self.features[qid][docid]["bm25"] = 0.0
            return
        
        bm25_score = 0.0
        doc_term_counts = Counter(doc_terms)
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF: log(N/df) where df is document frequency
                # We'll use a simple approximation here
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                # BM25 formula
                numerator = (k1 + 1) * tf
                denominator = k1 * (1 - b + b * (doc_len / self.avg_doc_len)) + tf
                bm25_score += idf * (numerator / denominator)
        
        self.features[qid][docid]["bm25"] = bm25_score
    
    def compute_query_term_coverage(self, qid, docid, **args):
        """Number of query terms found in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = set(self.documents.index[docid])
        
        coverage = sum(1 for term in query_terms if term in doc_terms)
        self.features[qid][docid]["query_term_coverage"] = coverage
    
    def compute_query_term_coverage_ratio(self, qid, docid, **args):
        """Ratio of query terms in the document to the total number of query terms."""
        query_terms = self.queries.qmap[qid]
        if not query_terms:
            self.features[qid][docid]["query_term_coverage_ratio"] = 0.0
            return
            
        doc_terms = set(self.documents.index[docid])
        
        coverage = sum(1 for term in query_terms if term in doc_terms)
        ratio = coverage / len(query_terms)
        self.features[qid][docid]["query_term_coverage_ratio"] = ratio
    
    def compute_stream_length(self, qid, docid, **args):
        """Length of the document (number of terms)."""
        doc_len = self.documents.dl[docid]
        self.features[qid][docid]["stream_length"] = doc_len
    
    def compute_idf(self, qid, docid, **args):
        """Sum of inverse document frequencies for query terms found in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = set(self.documents.index[docid])
        
        idf_sum = 0.0
        for term in query_terms:
            if term in doc_terms:
                # Calculate IDF: log(N/df) where df is document frequency
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                idf_sum += idf
        
        self.features[qid][docid]["idf"] = idf_sum
    
    def compute_sum_stream_length_normalized_tf(self, qid, docid, **args):
        """Sum of term frequencies normalized by document length."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = max(1, self.documents.dl[docid])  # Avoid division by zero
        
        doc_term_counts = Counter(doc_terms)
        
        sum_norm_tf = sum(doc_term_counts[term] / doc_len 
                           for term in query_terms if term in doc_term_counts)
        
        self.features[qid][docid]["sum_stream_length_normalized_tf"] = sum_norm_tf
    
    def compute_min_stream_length_normalized_tf(self, qid, docid, **args):
        """Minimum normalized term frequency among query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = max(1, self.documents.dl[docid])  # Avoid division by zero
        
        doc_term_counts = Counter(doc_terms)
        
        # Get normalized tf values for query terms found in document
        norm_tfs = [doc_term_counts[term] / doc_len 
                     for term in query_terms if term in doc_term_counts]
        
        min_norm_tf = min(norm_tfs) if norm_tfs else 0.0
        self.features[qid][docid]["min_stream_length_normalized_tf"] = min_norm_tf
    
    def compute_max_stream_length_normalized_tf(self, qid, docid, **args):
        """Maximum normalized term frequency among query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = max(1, self.documents.dl[docid])  # Avoid division by zero
        
        doc_term_counts = Counter(doc_terms)
        
        # Get normalized tf values for query terms found in document
        norm_tfs = [doc_term_counts[term] / doc_len 
                     for term in query_terms if term in doc_term_counts]
        
        max_norm_tf = max(norm_tfs) if norm_tfs else 0.0
        self.features[qid][docid]["max_stream_length_normalized_tf"] = max_norm_tf
    
    def compute_mean_stream_length_normalized_tf(self, qid, docid, **args):
        """Mean normalized term frequency of query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = max(1, self.documents.dl[docid])  # Avoid division by zero
        
        doc_term_counts = Counter(doc_terms)
        
        # Get normalized tf values for query terms found in document
        norm_tfs = [doc_term_counts[term] / doc_len 
                     for term in query_terms if term in doc_term_counts]
        
        mean_norm_tf = np.mean(norm_tfs) if norm_tfs else 0.0
        self.features[qid][docid]["mean_stream_length_normalized_tf"] = mean_norm_tf
    
    def compute_var_stream_length_normalized_tf(self, qid, docid, **args):
        """Variance of normalized term frequencies of query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        doc_len = max(1, self.documents.dl[docid])  # Avoid division by zero
        
        doc_term_counts = Counter(doc_terms)
        
        # Get normalized tf values for query terms found in document
        norm_tfs = [doc_term_counts[term] / doc_len 
                     for term in query_terms if term in doc_term_counts]
        
        var_norm_tf = np.var(norm_tfs) if len(norm_tfs) > 1 else 0.0
        self.features[qid][docid]["var_stream_length_normalized_tf"] = var_norm_tf
    
    def compute_sum_tfidf(self, qid, docid, **args):
        """Sum of TF-IDF scores for query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        
        doc_term_counts = Counter(doc_terms)
        sum_tfidf = 0.0
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                sum_tfidf += tf * idf
        
        self.features[qid][docid]["sum_tfidf"] = sum_tfidf
    
    def compute_min_tfidf(self, qid, docid, **args):
        """Minimum TF-IDF score among query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        
        doc_term_counts = Counter(doc_terms)
        tfidf_scores = []
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                tfidf_scores.append(tf * idf)
        
        min_tfidf = min(tfidf_scores) if tfidf_scores else 0.0
        self.features[qid][docid]["min_tfidf"] = min_tfidf
    
    def compute_max_tfidf(self, qid, docid, **args):
        """Maximum TF-IDF score among query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        
        doc_term_counts = Counter(doc_terms)
        tfidf_scores = []
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                tfidf_scores.append(tf * idf)
        
        max_tfidf = max(tfidf_scores) if tfidf_scores else 0.0
        self.features[qid][docid]["max_tfidf"] = max_tfidf
    
    def compute_mean_tfidf(self, qid, docid, **args):
        """Mean TF-IDF score of query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        
        doc_term_counts = Counter(doc_terms)
        tfidf_scores = []
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                tfidf_scores.append(tf * idf)
        
        mean_tfidf = np.mean(tfidf_scores) if tfidf_scores else 0.0
        self.features[qid][docid]["mean_tfidf"] = mean_tfidf
    
    def compute_var_tfidf(self, qid, docid, **args):
        """Variance of TF-IDF scores of query terms in the document."""
        query_terms = self.queries.qmap[qid]
        doc_terms = self.documents.index[docid]
        
        doc_term_counts = Counter(doc_terms)
        tfidf_scores = []
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term]
                
                # Calculate IDF
                df = sum(1 for doc_contents in self.documents.index.values() 
                          if term in doc_contents)
                df = max(1, df)  # Avoid division by zero
                idf = np.log(self.documents.num_docs / df)
                
                tfidf_scores.append(tf * idf)
        
        var_tfidf = np.var(tfidf_scores) if len(tfidf_scores) > 1 else 0.0
        self.features[qid][docid]["var_tfidf"] = var_tfidf
        # END SOLUTION

class GenerateFeatures:
    def __init__(self, feature_extractor: FeatureExtraction) -> None:
        self.fe = feature_extractor

    def run(self, qdr_path: str, qdr_feature_path: str, **fe_args):
        with open(qdr_feature_path, "w") as qdr_feature_file:
            with open(qdr_path, "r") as qdr_file:
                for line in tqdm(qdr_file):
                    qid, docid, rel = line.strip().split("\t")
                    features = self.fe.extract(qid, docid, **fe_args)
                    feature_line = "{} qid:{} {}\n".format(
                        rel,
                        qid,
                        " ".join(
                            "" if f is None else "{}:{}".format(i, f)
                            for i, f in enumerate(features.values())
                        ),
                    )

                    qdr_feature_file.write(feature_line)


class DataSet(object):
    """
    Class designed to manage meta-data for datasets.
    """

    def __init__(
        self,
        name,
        data_paths,
        num_rel_labels,
        num_features,
        num_nonzero_feat,
        feature_normalization=True,
        purge_test_set=True,
    ):
        self.name = name
        self.num_rel_labels = num_rel_labels
        self.num_features = num_features
        self.data_paths = data_paths
        self.purge_test_set = purge_test_set
        self._num_nonzero_feat = num_nonzero_feat

    def num_folds(self):
        return len(self.data_paths)

    def get_data_folds(self):
        return [DataFold(self, i, path) for i, path in enumerate(self.data_paths)]


class DataFoldSplit(object):
    def __init__(self, datafold, name, doclist_ranges, feature_matrix, label_vector):
        self.datafold = datafold
        self.name = name
        self.doclist_ranges = doclist_ranges
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector

    def num_queries(self):
        return self.doclist_ranges.shape[0] - 1

    def num_docs(self):
        return self.feature_matrix.shape[0]

    def query_range(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return s_i, e_i

    def query_size(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return e_i - s_i

    def query_sizes(self):
        return self.doclist_ranges[1:] - self.doclist_ranges[:-1]

    def query_labels(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.label_vector[s_i:e_i]

    def query_feat(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i:e_i, :]

    def doc_feat(self, query_index, doc_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        assert s_i + doc_index < self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i + doc_index, :]

    def doc_str(self, query_index, doc_index):
        doc_feat = self.doc_feat(query_index, doc_index)
        feat_i = np.where(doc_feat)[0]
        doc_str = ""
        for f_i in feat_i:
            doc_str += "%f " % (doc_feat[f_i])
        return doc_str

    def subsample_by_ids(self, qids):
        feature_matrix = []
        label_vector = []
        doclist_ranges = [0]
        for qid in qids:
            feature_matrix.append(self.query_feat(qid))
            label_vector.append(self.query_labels(qid))
            doclist_ranges.append(self.query_size(qid))

        doclist_ranges = np.cumsum(np.array(doclist_ranges), axis=0)
        feature_matrix = np.concatenate(feature_matrix, axis=0)
        label_vector = np.concatenate(label_vector, axis=0)
        return doclist_ranges, feature_matrix, label_vector

    def random_subsample(self, subsample_size):
        if subsample_size > self.num_queries():
            return DataFoldSplit(
                self.datafold,
                self.name + "_*",
                self.doclist_ranges,
                self.feature_matrix,
                self.label_vector,
                self.data_raw_path,
            )
        qids = np.random.randint(0, self.num_queries(), subsample_size)

        doclist_ranges, feature_matrix, label_vector = self.subsample_by_ids(qids)

        return DataFoldSplit(
            None, self.name + str(qids), doclist_ranges, feature_matrix, label_vector
        )


class DataFold(object):
    def __init__(self, dataset, fold_num, data_path):
        self.name = dataset.name
        self.num_rel_labels = dataset.num_rel_labels
        self.num_features = dataset.num_features
        self.fold_num = fold_num
        self.data_path = data_path
        self._data_ready = False
        self._num_nonzero_feat = dataset._num_nonzero_feat

    def data_ready(self):
        return self._data_ready

    def clean_data(self):
        del self.train
        del self.validation
        del self.test
        self._data_ready = False
        gc.collect()

    def read_data(self):
        """
        Reads data from a fold folder (letor format).
        """

        output = load_data_in_libsvm_format(
            self.data_path + "train_pairs_graded.tsvg", feature_size=self.num_features
        )
        train_feature_matrix, train_label_vector, train_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        output = load_data_in_libsvm_format(
            self.data_path + "dev_pairs_graded.tsvg", feature_size=self.num_features
        )
        valid_feature_matrix, valid_label_vector, valid_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        output = load_data_in_libsvm_format(
            self.data_path + "test_pairs_graded.tsvg", feature_size=self.num_features
        )
        test_feature_matrix, test_label_vector, test_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        self.train = DataFoldSplit(
            self,
            "train",
            train_doclist_ranges,
            train_feature_matrix,
            train_label_vector,
        )
        self.validation = DataFoldSplit(
            self,
            "validation",
            valid_doclist_ranges,
            valid_feature_matrix,
            valid_label_vector,
        )
        self.test = DataFoldSplit(
            self, "test", test_doclist_ranges, test_feature_matrix, test_label_vector
        )

        self.grading = self.train.random_subsample(1)

        self._data_ready = True


# this is a useful class to create torch DataLoaders, and can be used during training
class LTRData(Dataset):
    def __init__(self, data, split):
        split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
            "grading": data.grading,
        }.get(split)
        assert split is not None, "Invalid split!"
        features, labels = split.feature_matrix, split.label_vector
        self.doclist_ranges = split.doclist_ranges
        self.num_queries = split.num_queries()
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


class QueryGroupedLTRData(Dataset):
    def __init__(self, data, split):
        self.split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
            "grading": data.grading,
        }.get(split)
        assert self.split is not None, "Invalid split!"

    def __len__(self):
        return self.split.num_queries()

    def __getitem__(self, q_i):
        feature = torch.FloatTensor(self.split.query_feat(q_i))
        labels = torch.FloatTensor(self.split.query_labels(q_i))
        return feature, labels


# the return types are different from what pytorch expects,
# so we will define a custom collate function which takes in
# a batch and returns tensors (qids, features, labels)
def qg_collate_fn(batch):

    # qids = []
    features = []
    labels = []

    for f, l in batch:
        # qids.append(1)
        features.append(f)
        labels.append(l)

    return features, labels


def load_data():
    fold_paths = ["./data/"]
    num_relevance_labels = 5
    num_nonzero_feat = 15
    num_unique_feat = 15
    data = DataSet(
        "ir1-2023", fold_paths, num_relevance_labels, num_unique_feat, num_nonzero_feat
    )

    data = data.get_data_folds()[0]
    data.read_data()
    return data


# TODO: Implement this
class ClickLTRData(Dataset):
    def __init__(self, data, logging_policy):
        self.split = data.train
        self.logging_policy = logging_policy

    def __len__(self):
        return self.split.num_queries()

    def __getitem__(self, q_i):
        clicks = self.logging_policy.gather_clicks(q_i)
        positions = self.logging_policy.query_positions(q_i)

        ### BEGIN SOLUTION
        # Select only positions less than 20 (focusing on top results)
        valid_indices = [i for i, pos in enumerate(positions) if pos < 20]
        
        # Filter features, clicks, and positions based on valid indices
        features = self.split.query_feat(q_i)
        filtered_features = [features[positions[i]] for i in valid_indices]
        filtered_clicks = [clicks[i] for i in valid_indices]
        filtered_positions = [positions[i] for i in valid_indices]
        
        # Convert to tensors
        tensor_features = torch.FloatTensor(filtered_features)
        tensor_clicks = torch.FloatTensor(filtered_clicks)
        tensor_positions = torch.LongTensor(filtered_positions)
        ### END SOLUTION

        return tensor_features, tensor_clicks, tensor_positions

import pickle

if __name__ == "__main__":
    QUERIES_PATH = "./data/queries.tsv"
    STOP_WORDS_PATH = "./data/common_words"
    COLLECTION_PATH = "./data/collection.tsv"
    DOC_JSON = "./datasets/doc.pickle"

    print("Starting Preprocessing...")

    prp = Preprocess(STOP_WORDS_PATH)

    queries = Queries(prp)
    queries.preprocess_queries(QUERIES_PATH)

    print("Preprocessing Queries Complete")

    documents = Documents(prp)

    print("Preprocessing Documents...")

    documents.process_documents(COLLECTION_PATH)

    with open(DOC_JSON, "wb") as file:
        pickle.dump(documents, file)