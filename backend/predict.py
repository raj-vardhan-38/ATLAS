# =============================================================================
# ATLAS - SELF-CONTAINED PREDICTION SCRIPT
# =============================================================================
# This script has been refactored to combine the logic of the main predictor
# and the entire 'Explorer' pipeline into a single, self-contained file.
#
# Changes from the original:
#   - Subprocess calls to the explorer scripts have been replaced with direct
#     function calls to new, internal functions.
#   - Temporary file creation for passing data between explorer steps has
#     been removed. All data is now passed in-memory.
#   - All necessary imports for all parts of the pipeline are now at the top.
#
# This file serves as a single, runnable unit, which is a critical first
# step towards creating a standalone, installable application.
# =============================================================================

# --- Core Imports ---
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import pickle
import sys
from collections import Counter
from tqdm import tqdm
import uuid
import tensorflow as tf
import gc
from huggingface_hub import hf_hub_download
import os

# --- Imports for the now-internal Explorer pipeline ---
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import hdbscan
    from sklearn.preprocessing import normalize
    from Bio.Blast import NCBIWWW
    from Bio.Blast import NCBIXML
    import io
    EXPLORER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Explorer dependencies not available: {e}")
    EXPLORER_AVAILABLE = False

# --- Configuration ---
# Hugging Face Model Repository Configuration
MODEL_REPO = "aashutosh69/Atlas"
MODEL_FILES = {
    "16s": {
        "model": "16s_genus_classifier.h5",
        "vectorizer": "16s_genus_vectorizer.pkl", 
        "encoder": "16s_genus_label_encoder.pkl"
    },
    "18s": {
        "model": "18s_genus_classifier.h5",
        "vectorizer": "18s_genus_vectorizer.pkl",
        "encoder": "18s_genus_label_encoder.pkl"
    },
    "coi": {
        "model": "coi_genus_classifier.h5", 
        "vectorizer": "coi_genus_vectorizer.pkl",
        "encoder": "coi_genus_label_encoder.pkl"
    },
    "its": {
        "model": "its_genus_classifier.h5",
        "vectorizer": "its_genus_vectorizer.pkl", 
        "encoder": "its_genus_label_encoder.pkl"
    }
}

# Local paths
project_root = Path(__file__).parent
MODELS_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Explorer Pipeline Parameters (moved here from original scripts) ---
EXPLORER_KMER_SIZE = 6
EXPLORER_VECTOR_SIZE = 100

# --- Helper Function for K-mer Counting ---
def get_kmer_counts(sequence, k):
    """Calculates k-mer counts for a single DNA sequence."""
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if "N" not in kmer.upper():
            counts[kmer] += 1
    return dict(counts)

# --- Classifier Class ---
class TaxonClassifier:
    """A wrapper class to hold a trained model and its associated artifacts."""
    def __init__(self, marker_name, kmer_size):
        self.name = marker_name
        self.kmer_size = kmer_size
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = self._load_artifacts()

    def _load_artifacts(self):
        """Downloads and loads the model, vectorizer, and label encoder from Hugging Face."""
        try:
            if self.name not in MODEL_FILES:
                print(f"Warning: No model configuration found for {self.name}. Skipping this classifier.")
                return False
            
            model_config = MODEL_FILES[self.name]
            
            # Download model files from Hugging Face if they don't exist locally
            model_path = self._download_model_file(model_config["model"])
            vectorizer_path = self._download_model_file(model_config["vectorizer"])
            encoder_path = self._download_model_file(model_config["encoder"])
            
            if not all(p and Path(p).exists() for p in [model_path, vectorizer_path, encoder_path]):
                print(f"Warning: Failed to download one or more artifacts for {self.name}. Skipping this classifier.")
                return False
            
            print(f"  - Loading {self.name} model from {model_path}")
            
            # Try multiple loading approaches for better compatibility
            try:
                # First try: Import keras separately for compatibility
                from tensorflow import keras
                self.model = keras.models.load_model(str(model_path), compile=False)
            except Exception as e1:
                try:
                    # Second try: Use tf.keras if available
                    self.model = tf.keras.models.load_model(str(model_path), compile=False)
                except Exception as e2:
                    try:
                        # Third try: Import keras directly
                        import keras
                        self.model = keras.models.load_model(str(model_path), compile=False)
                    except Exception as e3:
                        try:
                            # Fourth try: Try loading with different file extension
                            h5_path = str(model_path).replace('.keras', '.h5')
                            if Path(h5_path).exists():
                                from tensorflow import keras
                                self.model = keras.models.load_model(h5_path, compile=False)
                            else:
                                raise Exception(f"No alternative model format found")
                        except Exception as e4:
                            raise Exception(f"All loading methods failed. Original error: {e1}, Final error: {e4}")
            
            # Load vectorizer and encoder
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print(f"  - Successfully loaded {self.name} model")
            return True
        except Exception as e:
            print(f"Error loading {self.name} model: {e}")
            return False

    def _download_model_file(self, filename):
        """Downloads a model file from Hugging Face Hub if it doesn't exist locally."""
        local_path = MODELS_DIR / filename
        
        # Force download from HuggingFace (ignore local cache)
        print(f"    - Downloading {filename} from Hugging Face...")
        
        try:
            # Get token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                print(f"    - WARNING: No HUGGINGFACE_TOKEN found in environment variables")
                print(f"    - Attempting download without authentication...")
            else:
                print(f"    - Using HUGGINGFACE_TOKEN from environment (length: {len(hf_token)})")
            
            # Try download with authentication
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                token=hf_token,
                force_download=False  # Allow caching for faster loads
            )
            print(f"    - Successfully downloaded {filename} from HuggingFace")
            return downloaded_path
            
        except Exception as hf_error:
            print(f"    - HuggingFace download failed: {hf_error}")
            
            # Try without token as fallback
            try:
                print(f"    - Retrying download without token...")
                downloaded_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    force_download=False
                )
                print(f"    - Successfully downloaded {filename} without token")
                return downloaded_path
            except Exception as no_token_error:
                print(f"    - Download without token also failed: {no_token_error}")
            
            # Final fallback to local file if it exists
            if local_path.exists():
                print(f"    - Using local cached {filename}")
                return str(local_path)
            else:
                print(f"    - No local file found for {filename}")
                return None
        

    def predict(self, sequence, confidence_threshold=0.8):
        """Predicts the taxon for a single sequence."""
        if not self.is_loaded:
            return None, 0.0
            
        kmer_counts = get_kmer_counts(sequence, self.kmer_size)
        
        if not kmer_counts:
            return None, 0.0

        vectorized_sequence = self.vectorizer.transform([kmer_counts])
        
        prediction_probabilities = self.model.predict(vectorized_sequence, verbose=0)[0]
        
        top_prob = np.max(prediction_probabilities)
        top_class_index = np.argmax(prediction_probabilities)
        
        if top_prob >= confidence_threshold:
            predicted_label = self.label_encoder.inverse_transform([top_class_index])[0]
            return predicted_label, top_prob
        
        return None, top_prob

# --- GPU Check Function ---
def check_gpu_status():
    """Checks and returns the GPU availability status."""
    try:
        # Try the newer TensorFlow 2.x API first
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"GPU is available and configured: {gpus[0].name}"
        else:
            return "GPU not found. Running on CPU."
    except AttributeError:
        # Fallback for older TensorFlow versions
        try:
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
            if gpu_devices:
                return f"GPU is available: {gpu_devices[0].name}"
            else:
                return "GPU not found. Running on CPU."
        except Exception:
            return "GPU status unknown. Running on CPU."

# =============================================================================
# --- Explorer Pipeline Logic (Moved from separate scripts) ---
# =============================================================================

def explorer_step_1_vectorize(sequences):
    """
    (Refactored from 01_vectorize_sequences.py)
    Vectorizes unclassified sequences using Doc2Vec downloaded from Hugging Face.
    """
    if not EXPLORER_AVAILABLE:
        print("  - [WARNING] Explorer dependencies not available. Skipping vectorization.")
        return None, None
        
    if not sequences:
        return None, None
        
    # Download Doc2Vec model from Hugging Face if not available locally
    doc2vec_model_path = _download_explorer_model("explorer_doc2vec.model")
    
    if not doc2vec_model_path or not Path(doc2vec_model_path).exists():
        print("  - [ERROR] Doc2Vec model not found and could not be downloaded. Cannot run Explorer pipeline.")
        return None, None

    print("  - Step 1.1: Loading Doc2Vec model...")
    try:
        doc2vec_model = Doc2Vec.load(str(doc2vec_model_path))
    except Exception as e:
        print(f"  - [ERROR] Failed to load Doc2Vec model: {e}")
        return None, None
    
    print("  - Step 1.2: Extracting vectors from sequences...")
    # The original script trained a new model every time. This is more efficient.
    sequence_vectors = np.array([
        doc2vec_model.infer_vector(
            get_kmer_counts(str(seq.seq), EXPLORER_KMER_SIZE),
            epochs=20  # Use a fixed number of inference epochs
        )
        for seq in tqdm(sequences, desc="    - Inferring vectors")
    ])
    
    # It's good practice to normalize the vectors for clustering
    sequence_vectors = normalize(sequence_vectors)
    
    sequence_ids = np.array([seq.id for seq in sequences])
    
    return sequence_vectors, sequence_ids

def _download_explorer_model(filename):
    """Downloads the Explorer Doc2Vec model from Hugging Face Hub if it doesn't exist locally."""
    local_path = MODELS_DIR / filename
    
    # If file already exists locally, return the path
    if local_path.exists():
        print(f"    - Using cached {filename}")
        return str(local_path)
    
    try:
        print(f"    - Downloading {filename} from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            cache_dir=str(MODELS_DIR.parent / ".cache"),
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        print(f"    - Successfully downloaded {filename}")
        return downloaded_path
    except Exception as e:
        print(f"    - Failed to download {filename}: {e}")
        return None

def explorer_step_2_cluster(sequence_vectors, sequence_ids):
    """
    (Refactored from 02_cluster_sequences.py)
    Clusters sequence vectors using HDBSCAN.
    """
    if not EXPLORER_AVAILABLE:
        print("  - [WARNING] Explorer dependencies not available. Skipping clustering.")
        return None
        
    if sequence_vectors is None or sequence_ids is None or len(sequence_vectors) < 5:
        return None
        
    print("  - Step 2: Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(sequence_vectors)
    
    # Create an in-memory DataFrame with the results
    df_results = pd.DataFrame({
        'sequence_id': sequence_ids,
        'cluster_label': cluster_labels
    })
    
    return df_results

def explorer_step_3_interpret(df_clusters, all_sequences, sequence_vectors):
    """
    (Refactored from 03_interpret_clusters.py)
    Interprets clusters by finding representatives and running BLAST.
    """
    if df_clusters is None or not all_sequences or sequence_vectors is None:
        return "No significant clusters were discovered in the input data."

    print("  - Step 3: Interpreting clusters and running BLAST...")
    
    # Create a dictionary for fast sequence lookups
    sequences_dict = {rec.id: str(rec.seq) for rec in all_sequences}
    id_to_vector_index = {seq_id: i for i, seq_id in enumerate(df_clusters['sequence_id'])}
    
    report_lines = []
    unique_cluster_ids = sorted(df_clusters['cluster_label'].unique())
    if -1 in unique_cluster_ids:
        unique_cluster_ids.remove(-1) # Ignore the noise points

    for cluster_id in tqdm(unique_cluster_ids, desc="    - Analyzing clusters"):
        cluster_df = df_clusters[df_clusters['cluster_label'] == cluster_id]
        member_ids = cluster_df['sequence_id'].tolist()
        
        member_indices = [id_to_vector_index[seq_id] for seq_id in member_ids]
        cluster_vectors = sequence_vectors[member_indices]
        
        centroid = np.mean(cluster_vectors, axis=0)
        distances = [np.linalg.norm(vec - centroid) for vec in cluster_vectors]
        rep_index_in_cluster = np.argmin(distances)
        representative_id = member_ids[rep_index_in_cluster]
        representative_sequence = sequences_dict[representative_id]
        
        try:
            # We will perform the BLAST search here directly
            result_handle = NCBIWWW.qblast("blastn", "nt", representative_sequence)
            blast_record = NCBIXML.read(result_handle)
            
            top_hit_title = "No significant similarity found."
            if blast_record.alignments:
                top_hit_title = blast_record.alignments[0].title
                
        except Exception as e:
            top_hit_title = f"BLAST query failed: {e}"

        report_lines.append(f"Cluster ID: {cluster_id}")
        report_lines.append(f"  - Size: {len(member_ids)} sequences")
        report_lines.append(f"  - Representative Sequence ID: {representative_id}")
        report_lines.append(f"  - BLAST Hypothesis: {top_hit_title}\n")
    
    if report_lines:
        return "\n".join(report_lines)
    else:
        return "No significant clusters were discovered in the input data."


# =============================================================================
# --- Main Analysis Function ---
# =============================================================================
def run_analysis(input_fasta_path):
    """
    Main analysis pipeline, now fully self-contained.
    
    Args:
        input_fasta_path (str): Path to the input FASTA file.
        
    Returns:
        dict: A dictionary containing the analysis results.
    """
    # --- 1. Check GPU Status ---
    gpu_status = check_gpu_status()
    print(f"\n--- GPU Status: {gpu_status} ---")

    # --- 2. Load All Filter Models ---
    print("--- Step 1: Loading All 'Filter' AI Models ---")
    potential_classifiers = [
        TaxonClassifier("16s", 6),
        TaxonClassifier("18s", 6),
        TaxonClassifier("coi", 8),
        TaxonClassifier("its", 7)
    ]
    
    classifiers = [clf for clf in potential_classifiers if clf.is_loaded]
    
    if not classifiers:
        return {"error": "No trained models found. Please ensure models are available."}
    
    print(f"  - Successfully loaded {len(classifiers)} models: {[clf.name for clf in classifiers]}")
    if len(classifiers) < len(potential_classifiers):
        missing = [clf.name for clf in potential_classifiers if not clf.is_loaded]
        print(f"  - Warning: Missing models for: {missing}")
    
    # Clear memory just in case, for a clean run
    tf.keras.backend.clear_session()
    gc.collect()

    # --- 3. Process Input FASTA ---
    print(f"\n--- Step 2: Processing Input File: {Path(input_fasta_path).name} ---")
    try:
        input_sequences = list(SeqIO.parse(input_fasta_path, "fasta"))
    except Exception as e:
        return {"error": f"Failed to parse FASTA file: {e}"}
        
    classified_results = Counter()
    unclassified_sequences = []

    for seq_record in tqdm(input_sequences, desc="  - Classifying sequences"):
        sequence_str = str(seq_record.seq)
        prediction_made = False
        for classifier in classifiers:
            label, prob = classifier.predict(sequence_str)
            if label:
                classified_results[label] += 1
                prediction_made = True
                break
        
        if not prediction_made:
            unclassified_sequences.append(seq_record)

    print(f"  - Classification complete.")
    print(f"    - Known organisms identified: {sum(classified_results.values())}")
    print(f"    - Unclassified sequences: {len(unclassified_sequences)}")
    
    # --- 4. Run Explorer Pipeline (if necessary) ---
    explorer_report_content = "No unclassified sequences to explore."
    if unclassified_sequences:
        if EXPLORER_AVAILABLE:
            print("\n--- Step 3: Starting 'Explorer' AI Pipeline ---")
            
            # --- REFACTORED: Call internal functions instead of subprocesses ---
            # The data is passed directly between these function calls in memory.
            sequence_vectors, sequence_ids = explorer_step_1_vectorize(unclassified_sequences)
            df_clusters = explorer_step_2_cluster(sequence_vectors, sequence_ids)
            explorer_report_content = explorer_step_3_interpret(df_clusters, unclassified_sequences, sequence_vectors)

            print("  - Explorer pipeline complete.")
        else:
            print("\n--- Step 3: Explorer Pipeline Unavailable ---")
            explorer_report_content = f"Explorer pipeline dependencies not available. {len(unclassified_sequences)} sequences remain unclassified and require manual analysis."

    # --- 5. Generate Final Report ---
    print("\n--- Step 4: Generating Final Biodiversity Report ---")
    report_file_name = f"ATLAS_REPORT_{Path(input_fasta_path).stem}_{uuid.uuid4().hex}.txt"
    final_report_path = REPORTS_DIR / report_file_name
    
    with open(final_report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("       ATLAS: AI Taxonomic Learning & Analysis System\n")
        f.write("                         FINAL REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"GPU Status: {gpu_status}\n")
        f.write(f"Input File: {Path(input_fasta_path).name}\n")
        f.write(f"Total Sequences Analyzed: {len(input_sequences)}\n\n")

        f.write("-" * 30 + "\n")
        f.write("Part 1: Known Organisms (Filter Results)\n")
        f.write("-" * 30 + "\n\n")
        if classified_results:
            for genus, count in sorted(classified_results.items(), key=lambda item: item[1], reverse=True):
                f.write(f"- {genus}: {count} sequences\n")
        else:
            f.write("No known organisms were identified by the Filter models.\n")
        
        f.write("\n\n" + "-" * 30 + "\n")
        f.write("Part 2: Novel Taxa Discovery (Explorer Results)\n")
        f.write("-" * 30 + "\n\n")
        f.write(explorer_report_content)
    
    print(f"  - Report saved successfully to: {final_report_path}")

    # Return a summary for the UI
    return {
        "status": "success",
        "report_path": str(final_report_path),
        "total_sequences": len(input_sequences),
        "classified": sum(classified_results.values()),
        "unclassified": len(unclassified_sequences),
        "classified_results": {k: v for k, v in classified_results.items()},
        "explorer_report": explorer_report_content
    }

# =============================================================================
# --- Main execution block for command-line use ---
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS: AI Taxonomic Learning & Analysis System")
    parser.add_argument(
        '--input_fasta', type=Path, required=True,
        help="Path to the input FASTA file for analysis."
    )
    args = parser.parse_args()
    
    run_analysis(args.input_fasta)
    print("\n[SUCCESS] ATLAS analysis complete.")