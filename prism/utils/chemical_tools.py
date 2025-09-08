# ===================================================================
# PRISM-DDI Chemical Tools
#
# This file contains core utilities for chemical informatics processing,
# primarily adapted from the insightful early-version code of MeTDDI.
# It handles SMILES parsing, feature extraction, and molecule decomposition.
# ===================================================================
import json
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
# +---------------------------------------------------------+
# |          PART 1: ATOM FEATURE EXTRACTION                |
# |          (Source: AMIE's features.py)                   |
# +---------------------------------------------------------+
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set: {allowable_set}")
    return [x == s for s in allowable_set]
def atom_features(atom, explicit_H=False, use_chirality=True):
    """
    Generates a feature vector for a single atom.
    """
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
         'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
         'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + \
        one_of_k_encoding(atom.GetDegree(), list(range(11))) + \
        one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(7))) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'Other']) + \
        [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(5)))
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(results, dtype=np.float32)
# +---------------------------------------------------------+
# |      PART 2: MOLECULE TO MOTIF DECOMPOSITION            |
# |      (Source: AMIE's utils.py -> Mol_Tokenizer)         |
# +---------------------------------------------------------+
class MotifTokenizer:
    """
    Handles the decomposition of a molecule into motifs (cliques) and
    tokenizes them based on a predefined vocabulary.
    """
    def __init__(self, vocab_path=''):
        self.vocab = {}
        if vocab_path:
            try:
                with open(vocab_path, 'r') as f:
                    self.vocab = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Vocabulary file not found at {vocab_path}. Tokenizer will be empty.")
        self.vocab_size = len(self.vocab)
    def tokenize(self, smiles):
        """
        Main tokenization function for training/evaluation.
        Takes a SMILES string and returns a sequence of motif IDs and the motif graph edges.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Could not parse SMILES during tokenization: {smiles}")
        cliques_atom_indices, edges = self._tree_decomp(mol)
        
        motif_id_sequence = [self.vocab.get('<global>', 3)]  # Start with global token
        
        for atom_indices in cliques_atom_indices:
            _, motif_smiles = self._get_clique_mol(mol, atom_indices)
            motif_id = self.vocab.get(motif_smiles, self.vocab.get('<unk>', 1))
            motif_id_sequence.append(motif_id)
            
        return motif_id_sequence, edges, cliques_atom_indices
    def tokenize_for_vocab(self, smiles):
        """
        A simplified tokenizer variant specifically for building the vocabulary.
        It only performs decomposition and returns the raw motif SMILES.
        This is the NEWLY ADDED method.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Warning: Could not parse SMILES for vocab building: {smiles}")
            return [], [], []
        cliques_atom_indices, edges = self._tree_decomp(mol)
        
        motif_smiles_list = []
        for atom_indices in cliques_atom_indices:
            _, motif_smiles = self._get_clique_mol(mol, atom_indices)
            if motif_smiles: # Ensure the motif is valid
                motif_smiles_list.append(motif_smiles)
        return motif_smiles_list, edges, cliques_atom_indices
    def edges_to_adj_matrix(self, num_nodes, edges):
        """Converts an edge list to a full adjacency matrix including the global token."""
        adj_matrix = np.eye(num_nodes, dtype=np.float32)
        for u, v in edges:
            # +1 to account for <global> token at index 0, as edges are 0-indexed for motifs
            idx1, idx2 = u + 1, v + 1
            if idx1 < num_nodes and idx2 < num_nodes:
                adj_matrix[idx1, idx2] = 1.0
                adj_matrix[idx2, idx1] = 1.0
        return adj_matrix
    def _get_clique_mol(self, mol, atoms_ids):
        """Extracts a molecule fragment (clique) and returns its canonical SMILES."""
        try:
            smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=True)
            new_mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if new_mol is None: # Sanitization failed
                return None, smiles
            canonical_smiles = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
            return new_mol, canonical_smiles
        except Exception:
            return None, ""
    def _tree_decomp(self, mol):
        """
        Performs tree decomposition of a molecule.
        (Logic adapted directly from MeTDDI's tree_decomp)
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []
        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        nei_list = [[] for i in range(n_atoms)]
        for i, clique in enumerate(cliques):
            for atom in clique:
                nei_list[atom].append(i)
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []
        
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)] # Rebuild nei_list
        for i, clique in enumerate(cliques):
            for atom in clique:
                nei_list[atom].append(i)
        
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 2]
            
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)
        
        edge_list = [u + (100 - v,) for u, v in edges.items()]
        if not edge_list: return cliques, []
        row, col, data = zip(*edge_list)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        mst_edges = [(r, c) for r, c in zip(row, col)]
        return cliques, mst_edges
# +---------------------------------------------------------+
# |   PART 3: FULL DRUG REPRESENTATION GENERATOR            |
# |   (Source: AMIE's dataset_processed.py -> molgraph_rep)   |
# +---------------------------------------------------------+
def generate_drug_representation(smiles, motif_cliques_atoms):
    """
    Generates the complete atomic representation for a single drug,
    including the bridge matrix to motifs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    # 1. Generate Atom Features
    atom_feature_matrix = np.array([atom_features(atom) for atom in mol.GetAtoms()])
    # 2. Generate Atom Adjacency Matrix
    adj_matrix = Chem.GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
    
    # 3. Generate Atom-to-Motif Bridge Matrix
    num_atoms = mol.GetNumAtoms()
    num_motifs = len(motif_cliques_atoms)
    atom_motif_matrix = np.zeros((num_motifs, num_atoms), dtype=np.float32)
    for i, atom_indices in enumerate(motif_cliques_atoms):
        for atom_idx in atom_indices:
            atom_motif_matrix[i, atom_idx] = 1.0
    # 4. Generate Sum of Atoms per Motif (for normalization)
    sum_atoms_per_motif = np.sum(atom_motif_matrix, axis=1, keepdims=True)
    sum_atoms_per_motif[sum_atoms_per_motif == 0] = 1.0 # Avoid division by zero
    return {
        'atomic_features': atom_feature_matrix.astype(np.float32),
        'atomic_adj': adj_matrix.astype(np.float32),
        'atom_motif_matrix': atom_motif_matrix,
        'sum_atoms_per_motif': sum_atoms_per_motif.astype(np.float32)
    }