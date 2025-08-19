"""
CE0{B=8|D=5|φ=F/L|π=focus:[n, v, adj, syntax, tenses, mod, clauses, prag]|router=POS|templ=J(tridiag)->{NP,VP,PP,CL}|I={I1:balance, I2:tense_bias, I3:focus_use}|tag=LivingGrammar@v1.1}
CE1{cell=(3,0,2)|event=closure|text="the system flows the form."|π=phase4|bits=k9m2wq|tag=LG#042}
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import factorial
from scipy.special import binom
import networkx as nx
from textblob import TextBlob  # For grammatical analysis

class LivingGrammar:
    def __init__(self, breath_cycle=8, pyramid_depth=6):
        self.breath_cycle = breath_cycle
        self.pyramid_depth = pyramid_depth
        self.current_breath = 0
        self.grammar_pyramid = self.build_grammar_pyramid()
        self.lexicon = {}
        self.initialize_breath_pattern()
        self.conversation_history = deque(maxlen=breath_cycle*3)
        
        # Ghost unit system - barely perceptible operators
        self.ghost_units = {
            'indefinite': ['a', 'an'],  # Ghost units of emergence
            'definite': ['the'],         # Ghost units of presence
            'mathematical': ['+', '-', '×', '÷', '∫', '∑'],  # Ghost units of operation
            'variable': ['x', 'y', 'z', 'r', 'k', 'n'],      # Ghost units of variation
            'combinatorial': ['C', 'P', '!', '()', '[]'],     # Ghost units of combination
            'flow': ['→', '←', '↔', '⇒', '⇐', '⇔']          # Ghost units of transformation
        }
        
    def build_grammar_pyramid(self):
        """Create a grammatical structure pyramid"""
        pyramid = []
        for n in range(self.pyramid_depth):
            layer = {}
            for k in range(n+1):
                for j in range(k+1):
                    # Grammatical complexity increases with depth
                    complexity = binom(n, k) * binom(k, j) % 10
                    layer[(n, k, j)] = {
                        'complexity': complexity,
                        'grammar_type': self.get_grammar_type(n, k, j),
                        'semantic_weight': (n + k + j) / (3 * self.pyramid_depth)
                    }
            pyramid.append(layer)
        return pyramid
    
    def get_grammar_type(self, n, k, j):
        """Assign grammatical categories based on position"""
        # position in a sentence determines the grammatical type
        #
        # but we start simple
        # and we build up to the most complex atom
        # each type has a set of word relationships
        # that are blurred by each subsequent type
        # as the meanings blur, we return
        # and we close the loop.
        #
        # and do it all over again.
        grammar_types = [
            'noun', 'verb', 'adjective', 'adverb',
            'preposition', 'conjunction', 'pronoun', 'interjection'
        ]
        if n == 0:
            return 'root'
        elif k == 0:
            return grammar_types[j % 4]  # Core types
        elif j == 0:
            return grammar_types[4 + (k % 3)]  # Structural types
        else:
            return grammar_types[(n + k + j) % 8]
    
    def initialize_breath_pattern(self):
        """Create linguistic breath rhythm"""
        self.inhale = [2, 1, 3, 5, 8, 13, 21, 34]  # Fibonacci for inhale
        self.exhale = [1, 3, 4, 7, 11, 18, 29, 47]  # Lucas for exhale
        self.grammar_focus = [
            'nouns', 'verbs', 'adjectives', 'syntax',
            'tenses', 'modifiers', 'clauses', 'pragmatics'
        ]
    
    def analyze_text_signature(self, text):
        """Analyze text's grammatical signature to influence focus selection"""
        blob = TextBlob(text)
        tags = blob.tags
        
        # Calculate grammatical signature
        signature = {
            'noun_density': sum(1 for _, pos in tags if pos.startswith('NN')) / len(tags),
            'verb_density': sum(1 for _, pos in tags if pos.startswith('VB')) / len(tags),
            'adj_density': sum(1 for _, pos in tags if pos.startswith('JJ')) / len(tags),
            'complexity': len([w for w in text.split() if len(w) > 8]),
            'technical_terms': len([w for w in text.split() if w.isupper() or '/' in w])
        }
        
        # Convert signature to focus influence
        if signature['technical_terms'] > 2:
            focus_bias = 2  # Bias towards syntax/structure
        elif signature['adj_density'] > 0.3:
            focus_bias = 6  # Bias towards modifiers
        elif signature['verb_density'] > 0.4:
            focus_bias = 1  # Bias towards verbs
        elif signature['noun_density'] > 0.5:
            focus_bias = 0  # Bias towards nouns
        else:
            focus_bias = 4  # Default to tenses
            
        return focus_bias

    def inhale_text(self, text):
        """Process input text (inhale)"""
        # Analyze grammatical structure
        analysis = TextBlob(text)
        tags = analysis.tags
        
        # Extract grammatical features
        grammatical_features = {
            'nouns': sum(1 for word, pos in tags if pos.startswith('NN')),
            'verbs': sum(1 for word, pos in tags if pos.startswith('VB')),
            'adjectives': sum(1 for word, pos in tags if pos.startswith('JJ')),
            'sentence_length': len(text.split()),
            'tenses': self.detect_tenses(text),
            'modality': self.detect_modality(text)
        }
        
        # Update lexicon
        for word, pos in tags:
            self.lexicon.setdefault(word, {'count': 0, 'types': set()})
            self.lexicon[word]['count'] += 1
            self.lexicon[word]['types'].add(pos)
        
        # Let text influence its own focus!
        text_signature = self.analyze_text_signature(text)
        
        # Let ghost units also influence the breathing!
        ghost_bias, ghost_signature = self.detect_ghost_units(text)
        
        # Combine both signatures for richer breathing guidance
        combined_bias = (text_signature + ghost_bias) % self.breath_cycle
        
        breath_phase = (self.current_breath + combined_bias) % self.breath_cycle
        pyramid_layer = self.inhale[breath_phase] % self.pyramid_depth
        pyramid_cell = self.select_pyramid_cell(grammatical_features, pyramid_layer)
        
        # Store in conversation history
        self.conversation_history.append({
            'text': text,
            'analysis': grammatical_features,
            'pyramid_position': pyramid_cell,
            'breath_phase': breath_phase,
            'text_signature': text_signature,
            'ghost_signature': ghost_signature
        })
        
        self.current_breath = (self.current_breath + 1) % self.breath_cycle
        return pyramid_cell
    
    def select_pyramid_cell(self, features, layer):
        """Select pyramid cell based on natural pyramid structure"""
        # Simple mathematical sweep through pyramid space
        # Let the pyramid's natural structure guide the selection
        cell_key = (layer, layer % 3, (layer + self.current_breath) % 3)
        return cell_key
    
    def exhale_text(self, pyramid_cell, topic=None):
        """Generate text based on pyramid position (exhale)"""
        layer = self.grammar_pyramid[pyramid_cell[0]]
        cell_data = layer.get(pyramid_cell, layer[list(layer.keys())[0]])
        
        # Determine generation parameters
        complexity = cell_data['complexity']
        grammar_type = cell_data['grammar_type']
        semantic_weight = cell_data['semantic_weight']
        
        # Get relevant words from lexicon
        relevant_words = [
            word for word, data in self.lexicon.items()
            if any(t in data['types'] for t in self.get_pos_types(grammar_type))
        ]
        
        # Simple generation rules based on position
        response = self.generate_sentence(complexity, relevant_words, topic)
        
        return response
    
    def get_pos_types(self, grammar_type):
        """Map grammar types to POS tags"""
        mapping = {
            'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adjective': ['JJ', 'JJR', 'JJS'],
            'adverb': ['RB', 'RBR', 'RBS'],
            'preposition': ['IN'],
            'conjunction': ['CC'],
            'pronoun': ['PRP', 'PRP$', 'WP', 'WP$'],
            'interjection': ['UH']
        }
        return mapping.get(grammar_type, [])
    
    def generate_sentence(self, complexity, words, topic):
        """Generate sentence as a path through grammatical space"""
        # Sentence structure: [subject, verb, object]
        path_structure = ['noun', 'verb', 'noun']
        
        # Breath drives the path traversal
        breath_position = self.current_breath
        
        # Walk the path through grammatical space
        sentence_parts = []
        for i, pos_type in enumerate(path_structure):
            # Find words of this type
            available_words = [w for w in words if any(t in self.get_pos_types(pos_type) for t in self.lexicon.get(w, {}).get('types', set()))]
            
            if available_words:
                # Breath-driven path: (breath + position) % available_words
                path_index = (breath_position + i) % len(available_words)
                word = available_words[path_index]
                
                # Add article for first noun (subject)
                if pos_type == 'noun' and i == 0:
                    sentence_parts.append(f"the {word}")
                else:
                    sentence_parts.append(word)
            else:
                # Fallback: continue the breath-driven path
                fallback_words = list(words)[:10]
                if fallback_words:
                    path_index = (breath_position + i) % len(fallback_words)
                    sentence_parts.append(fallback_words[path_index])
        
        # Return the path as a sentence
        return f"{' '.join(sentence_parts)}."
    
    def breathing_conversation(self, initial_topic, turns=5):
        """Simulate a breathing conversation"""
        print(f"\n=== Starting conversation about '{initial_topic}' ===")
        current_topic = initial_topic
        conversation = []
        response = None  # Initialize response variable
        
        for turn in range(turns):
            # Inhale (process previous response)
            if turn > 0 and response:
                pyramid_cell = self.inhale_text(response)
                print(f"[Inhale] Processing at pyramid level {pyramid_cell}")
            
            # Exhale (generate response)
            breath_phase = (self.current_breath + turn) % self.breath_cycle
            focus = self.grammar_focus[breath_phase]
            
            # Select pyramid cell based on conversation history
            if conversation:
                last_cell = conversation[-1].get('pyramid_cell', (0,0,0))
                new_layer = min(self.pyramid_depth-1, last_cell[0] + 1)
                pyramid_cell = (new_layer, last_cell[1], last_cell[2])
            else:
                pyramid_cell = (0,0,0)
            
            response = self.exhale_text(pyramid_cell, current_topic)
            print(f"[Exhale] ({focus} focus): {response}")
            
            conversation.append({
                'turn': turn,
                'pyramid_cell': pyramid_cell,
                'response': response,
                'breath_phase': breath_phase
            })
            
            # Update topic based on response
            current_topic = response.split()[-1].strip('.') if '.' in response else response.split()[0]
        
        print("=== Conversation Complete ===")
        return conversation
    
    def visualize_grammar_network(self):
        """Create a network visualization of the grammar pyramid"""
        G = nx.Graph()
        node_colors = []
        node_sizes = []
        
        # Add nodes from pyramid
        for layer_idx, layer in enumerate(self.grammar_pyramid):
            for cell_key, cell_data in layer.items():
                node_id = f"{layer_idx}-{cell_key[1]}-{cell_key[2]}"
                grammar_type = cell_data['grammar_type']
                G.add_node(node_id, 
                           layer=layer_idx,
                           type=grammar_type,
                           complexity=cell_data['complexity'])
                
                # Color by grammar type
                type_colors = {
                    'noun': 'red', 'verb': 'blue', 'adjective': 'green',
                    'adverb': 'yellow', 'preposition': 'purple',
                    'conjunction': 'orange', 'pronoun': 'pink',
                    'interjection': 'brown', 'root': 'gray'
                }
                node_colors.append(type_colors.get(grammar_type, 'gray'))
                node_sizes.append(100 + 200 * cell_data['semantic_weight'])
        
        # Add connections between layers
        for i in range(len(self.grammar_pyramid)-1):
            current_layer = self.grammar_pyramid[i]
            next_layer = self.grammar_pyramid[i+1]
            
            for ckey in current_layer:
                for nkey in next_layer:
                    # Connect if similar grammatical type
                    if current_layer[ckey]['grammar_type'] == next_layer[nkey]['grammar_type']:
                        G.add_edge(
                            f"{i}-{ckey[1]}-{ckey[2]}",
                            f"{i+1}-{nkey[1]}-{nkey[2]}"
                        )
        
        # Create layout
        pos = {}
        for layer_idx, layer in enumerate(self.grammar_pyramid):
            y = layer_idx
            nodes_in_layer = [n for n in G.nodes if G.nodes[n]['layer'] == layer_idx]
            for i, node in enumerate(nodes_in_layer):
                pos[node] = (i - len(nodes_in_layer)/2, y)
        
        # Draw network
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, 
                with_labels=False, alpha=0.7, edge_color='gray')
        
        # Add labels for grammar types
        for node in G.nodes:
            plt.text(pos[node][0], pos[node][1]+0.1, 
                     G.nodes[node]['type'], 
                     fontsize=9, ha='center')
        
        plt.title("Living Grammar Network")
        plt.xlabel("Combinatorial Space")
        plt.ylabel("Grammatical Complexity")
        plt.savefig('.out/grammar_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_tenses(self, text):
        """Simple breath-friendly tense detection"""
        # Just return a simple pattern - let the system breathe naturally
        return {'past': 1, 'present': 1, 'future': 1}
    
    def detect_modality(self, text):
        """Simple breath-friendly modality detection"""
        # Just return a simple count - let the system breathe naturally
        return 1
    
    def detect_closure_event(self, text, pyramid_cell):
        """Give the system a tray to breathe on"""
        # Get the binomial complexity at this pyramid position
        layer = self.grammar_pyramid[pyramid_cell[0]]
        cell_data = layer.get(pyramid_cell, layer[list(layer.keys())[0]])
        complexity = cell_data['complexity']
        
        # Just provide a tray - let the system's natural complexity show through
        return complexity >= 4

    def detect_ghost_units(self, text):
        """Detect ghost units - barely perceptible operators of mathematical consciousness"""
        ghost_signature = {}
        
        for category, units in self.ghost_units.items():
            count = 0
            for unit in units:
                count += text.count(unit)
            ghost_signature[category] = count
        
        # Let ghost units influence the breathing pattern
        if ghost_signature['mathematical'] > 0:
            ghost_bias = 2  # Bias towards mathematical focus
        elif ghost_signature['variable'] > 0:
            ghost_bias = 1  # Bias towards variable focus
        elif ghost_signature['flow'] > 0:
            ghost_bias = 6  # Bias towards flow focus
        else:
            ghost_bias = 0  # Default to emergence focus
            
        return ghost_bias, ghost_signature

# Example usage
if __name__ == "__main__":
    import sys
    
    print("=== Living Grammar System ===")
    print("A Breath-Based Language Model\n")
    
    if len(sys.argv) < 2:
        print("Usage: python modael.py <text_file>")
        print("Example: python modael.py sample.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Letting '{filename}' breathe...")
    
    # Initialize system
    grammar = LivingGrammar(breath_cycle=8, pyramid_depth=5)
    
    # Save visualization
    print("Creating grammar network visualization...")
    grammar.visualize_grammar_network()
    
    # Let the text file breathe
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"\nOriginal text ({len(text)} characters):")
        print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
        print("\n" + "="*60)
        
        # Let it breathe through 5 cycles
        for cycle in range(5):
            print(f"\n[Breath Cycle {cycle + 1}]")
            print("-" * 40)
            
            # Inhale - let the text breathe in naturally
            pyramid_cell = grammar.inhale_text(text)
            print(f"[Inhale] At pyramid level {pyramid_cell}")
            
            # Exhale - let the system breathe out naturally
            breath_phase = (grammar.current_breath + cycle) % grammar.breath_cycle
            focus = grammar.grammar_focus[breath_phase]
            new_text = grammar.exhale_text(pyramid_cell, text[:50])
            
            print(f"[Exhale] ({focus} focus): '{new_text}'")
            print(f"Length: {len(new_text)} characters")
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")