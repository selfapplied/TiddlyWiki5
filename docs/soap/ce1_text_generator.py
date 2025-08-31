import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math
import random
import string

class TextSeedKernel:
    """A seed computer specialized for text generation"""
    
    def __init__(self, position, seed_value, text_domain):
        self.position = position
        self.seed_value = seed_value
        self.text_domain = text_domain
        
        # Each text seed defines its own linguistic parameters
        self.define_linguistic_kernels()
        self.define_semantic_cuts()
        self.define_rhythmic_timing()
        self.define_textual_shapes()
    
    def define_linguistic_kernels(self):
        """Define how this seed processes linguistic information"""
        s = self.seed_value
        
        # Phonetic kernels - how sounds are processed
        self.phonetic_kernels = {
            'vowel_frequency': s * 0.8 + 0.2,  # Vowel density
            'consonant_clusters': int(s * 5) + 1,  # Max consonant grouping
            'syllable_length': s * 3 + 1,  # Average syllables per word
            'stress_pattern': s * 2 - 1  # Stress rhythm (-1 to 1)
        }
        
        # Semantic kernels - meaning processing
        self.semantic_kernels = {
            'abstraction_level': s,  # Concrete (0) to abstract (1)
            'emotional_valence': s * 2 - 1,  # Negative (-1) to positive (1)
            'complexity_preference': s,  # Simple (0) to complex (1)
            'metaphor_tendency': s * 0.8 + 0.2  # Literal (0.2) to metaphorical (1.0)
        }
        
        # Syntactic kernels - grammar structure
        self.syntactic_kernels = {
            'sentence_length': int(s * 20) + 5,  # Words per sentence
            'clause_complexity': s,  # Simple (0) to compound (1)
            'punctuation_style': s,  # Minimal (0) to elaborate (1)
            'word_order_flexibility': s  # Fixed (0) to flexible (1)
        }
        
        # Vocabulary kernels - word choice
        self.vocabulary_kernels = {
            'formality_level': s,  # Informal (0) to formal (1)
            'technical_preference': s,  # Common (0) to technical (1)
            'archaic_tendency': s * 0.3,  # Modern (0) to archaic (0.3)
            'neologism_creativity': s * 0.5  # Standard (0) to creative (0.5)
        }
    
    def define_semantic_cuts(self):
        """Define semantic boundaries and transitions"""
        s = self.seed_value
        
        # Semantic discontinuities - where meaning shifts
        self.semantic_cuts = {
            'topic_shifts': s * 0.8 + 0.2,  # How often topics change
            'temporal_jumps': s * 0.6 + 0.1,  # Time sequence breaks
            'perspective_changes': s * 0.7 + 0.1,  # Viewpoint shifts
            'logical_breaks': s * 0.5 + 0.1  # Logical sequence disruptions
        }
        
        # Cut types for semantic transitions
        self.cut_types = [
            'metaphor',      # Metaphorical connection
            'juxtaposition', # Direct contrast
            'ellipsis',      # Omission/implied connection
            'paradox',       # Contradictory connection
            'synesthesia'    # Cross-sensory connection
        ]
    
    def define_rhythmic_timing(self):
        """Define the temporal rhythm of text generation"""
        s = self.seed_value
        
        # Temporal parameters for text flow
        self.rhythm_params = {
            'pace': s * 2 + 0.5,  # Slow (0.5) to fast (2.5)
            'pulse': s * 10 + 5,  # Rhythmic beats per sentence
            'pause_frequency': s * 0.8 + 0.1,  # Natural pause points
            'acceleration_tendency': s  # Speed up (0) or slow down (1)
        }
        
        # Rhythmic functions for text generation
        self.rhythm_functions = {
            'breathing': lambda t: np.sin(t * self.rhythm_params['pulse']),
            'heartbeat': lambda t: np.cos(t * self.rhythm_params['pulse'] * 2),
            'wave': lambda t: np.sin(t * self.rhythm_params['pulse'] * 0.5),
            'staccato': lambda t: np.sign(np.sin(t * self.rhythm_params['pulse'] * 3))
        }
    
    def define_textual_shapes(self):
        """Define the structural shapes of text"""
        s = self.seed_value
        
        # Text structure patterns
        self.text_shapes = {
            'paragraph_length': int(s * 8) + 2,  # Sentences per paragraph
            'narrative_arc': s,  # Linear (0) to circular (1)
            'repetition_pattern': s * 0.8 + 0.1,  # How much repetition
            'variation_degree': s  # Monotony (0) to variety (1)
        }
        
        # Structural templates this seed prefers
        self.structural_templates = [
            'linear_progression',    # A → B → C
            'circular_return',       # A → B → C → A
            'spiral_development',    # A → B → C → A' → B' → C'
            'fractal_nesting',       # A(B(C)) → A'(B'(C'))
            'wave_oscillation'       # A → B → A → B → A
        ]
    
    def generate_word(self, context, time=0.0):
        """Generate a single word based on this seed's parameters"""
        # Apply rhythmic timing
        rhythm_factor = self.rhythm_functions['breathing'](time)
        
        # Choose word characteristics based on seed parameters
        word_length = int(self.phonetic_kernels['syllable_length'] * (1 + rhythm_factor))
        formality = self.vocabulary_kernels['formality_level']
        
        # Generate word based on domain and parameters
        if self.text_domain == 'poetry':
            return self._generate_poetic_word(word_length, formality, rhythm_factor)
        elif self.text_domain == 'prose':
            return self._generate_prose_word(word_length, formality, rhythm_factor)
        elif self.text_domain == 'technical':
            return self._generate_technical_word(word_length, formality, rhythm_factor)
        else:  # 'creative'
            return self._generate_creative_word(word_length, formality, rhythm_factor)
    
    def _generate_poetic_word(self, length, formality, rhythm):
        """Generate a word suitable for poetry"""
        # Poetry words tend to be more evocative
        base_words = ['light', 'shadow', 'whisper', 'echo', 'dream', 'memory', 'silence', 'music']
        if formality > 0.7:
            base_words.extend(['luminous', 'ethereal', 'transcendent', 'sublime'])
        if rhythm > 0:
            base_words.extend(['flowing', 'dancing', 'singing', 'dancing'])
        
        word = random.choice(base_words)
        # Adjust length by adding/removing syllables
        while len(word) < length * 3:
            word += random.choice(['ing', 'ed', 'ly', 'ness', 'ful'])
        return word[:length * 4]  # Cap maximum length
    
    def _generate_prose_word(self, length, formality, rhythm):
        """Generate a word suitable for prose"""
        # Prose words are more functional
        base_words = ['the', 'and', 'but', 'when', 'where', 'how', 'what', 'why']
        if formality > 0.5:
            base_words.extend(['however', 'therefore', 'furthermore', 'nevertheless'])
        if length > 2:
            base_words.extend(['because', 'although', 'through', 'between'])
        
        return random.choice(base_words)
    
    def _generate_technical_word(self, length, formality, rhythm):
        """Generate a technical/scientific word"""
        # Technical vocabulary
        base_words = ['algorithm', 'function', 'parameter', 'variable', 'constant']
        if formality > 0.8:
            base_words.extend(['optimization', 'implementation', 'specification'])
        if length > 3:
            base_words.extend(['computational', 'mathematical', 'theoretical'])
        
        return random.choice(base_words)
    
    def _generate_creative_word(self, length, formality, rhythm):
        """Generate a creative/neologistic word"""
        # Creative word formation
        if self.vocabulary_kernels['neologism_creativity'] > 0.3:
            # Create new words by combining parts
            prefixes = ['un', 're', 'pre', 'post', 'anti', 'pro', 'hyper', 'meta']
            roots = ['think', 'feel', 'know', 'see', 'hear', 'touch', 'move', 'grow']
            suffixes = ['ing', 'ed', 'ly', 'ness', 'ful', 'able', 'tion', 'sion']
            
            if random.random() < 0.5:
                return random.choice(prefixes) + random.choice(roots)
            else:
                return random.choice(roots) + random.choice(suffixes)
        else:
            # Use existing creative words
            creative_words = ['serendipity', 'serenity', 'luminosity', 'eternity', 'infinity']
            return random.choice(creative_words)
    
    def generate_sentence(self, context, time=0.0):
        """Generate a complete sentence"""
        # Determine sentence length
        target_length = int(self.syntactic_kernels['sentence_length'] * 
                           (1 + 0.5 * np.sin(time * self.rhythm_params['pulse'])))
        
        # Generate words
        words = []
        for i in range(target_length):
            word = self.generate_word(context, time + i * 0.1)
            words.append(word)
        
        # Apply semantic cuts for variety
        if random.random() < self.semantic_cuts['topic_shifts']:
            cut_type = random.choice(self.cut_types)
            if cut_type == 'metaphor':
                words.insert(len(words)//2, 'like')
            elif cut_type == 'juxtaposition':
                words.insert(len(words)//2, 'yet')
            elif cut_type == 'ellipsis':
                words.insert(len(words)//2, '...')
        
        # Capitalize first word and add punctuation
        if words:
            words[0] = words[0].capitalize()
            if random.random() < self.syntactic_kernels['punctuation_style']:
                words.append(random.choice(['.', '!', '?', '...']))
            else:
                words.append('.')
        
        return ' '.join(words)
    
    def compute_textual_field(self, query_points, context, time=0.0):
        """Compute the textual influence field of this seed"""
        # Transform to seed's local coordinates
        relative_points = query_points - self.position
        distances = np.linalg.norm(relative_points, axis=-1)
        
        # Generate text based on position and time
        text_influence = np.zeros_like(distances)
        
        # Apply distance-based influence directly
        influence_mask = distances < self.text_shapes['paragraph_length'] * 0.5
        
        # Generate a single sentence for this seed's influence
        sentence = self.generate_sentence(context, time)
        sentence_influence = len(sentence) * 0.01 + self.seed_value * 0.1
        
        # Apply influence where mask is True
        text_influence[influence_mask] = sentence_influence
        
        # Apply distance falloff
        influence = np.exp(-distances / (self.text_shapes['paragraph_length'] * 0.3))
        
        return text_influence * influence

class CE1TextGenerator:
    """Text generation system using distributed seed computers"""
    
    def __init__(self, num_seeds=32, text_domain='creative'):
        self.num_seeds = num_seeds
        self.text_domain = text_domain
        
        # Generate text seed computers
        self.text_seeds = self.generate_text_seeds()
        
        # Text generation context
        self.context = {
            'topic': 'mathematical beauty',
            'style': text_domain,
            'mood': 'contemplative',
            'length': 'medium'
        }
    
    def generate_text_seeds(self):
        """Create a collection of text-generating seed computers"""
        seeds = []
        
        # Distribute seeds across different text generation roles
        for i in range(self.num_seeds):
            # Different distribution strategies for different text functions
            if i < self.num_seeds // 4:  # Opening seeds
                x = -1.5 + 3.0 * (i / (self.num_seeds // 4 - 1))
                y = -1.0
                role = 'opening'
            elif i < self.num_seeds // 2:  # Development seeds
                x = -1.5 + 3.0 * ((i - self.num_seeds // 4) / (self.num_seeds // 4 - 1))
                y = 0.0
                role = 'development'
            elif i < 3 * self.num_seeds // 4:  # Climax seeds
                x = -1.5 + 3.0 * ((i - self.num_seeds // 2) / (self.num_seeds // 4 - 1))
                y = 1.0
                role = 'climax'
            else:  # Conclusion seeds
                x = -1.5 + 3.0 * ((i - 3 * self.num_seeds // 4) / (self.num_seeds // 4 - 1))
                y = 1.5
                role = 'conclusion'
            
            position = np.array([x, y])
            seed_value = (i / self.num_seeds + random.random() * 0.1) % 1.0
            
            seeds.append(TextSeedKernel(position, seed_value, self.text_domain))
        
        return seeds
    
    def generate_text(self, target_length=100, time=0.0):
        """Generate text through collective seed computation"""
        print(f"Generating {self.text_domain} text with {len(self.text_seeds)} seed computers...")
        
        # Initialize text components
        sentences = []
        current_length = 0
        
        # Each seed contributes to the text generation
        seed_index = 0
        while current_length < target_length and len(sentences) < 20:
            seed = self.text_seeds[seed_index % len(self.text_seeds)]
            
            # Generate sentence from this seed
            sentence = seed.generate_sentence(self.context, time + seed_index * 0.1)
            sentences.append(sentence)
            current_length += len(sentence)
            
            seed_index += 1
            
            # Add paragraph breaks based on seed parameters
            if seed_index % 3 == 0 and random.random() < 0.3:
                sentences.append('\n')
        
        # Combine into final text
        final_text = ' '.join(sentences)
        
        # Apply global text processing
        final_text = self.apply_global_text_processing(final_text)
        
        return final_text
    
    def apply_global_text_processing(self, text):
        """Apply global text transformations"""
        # Capitalize sentences
        sentences = text.split('. ')
        processed_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Ensure proper capitalization
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                processed_sentences.append(sentence)
        
        # Rejoin with proper punctuation
        return '. '.join(processed_sentences) + '.'
    
    def generate_text_visualization(self, text, time=0.0):
        """Create a visual representation of the text generation process"""
        # Create coordinate grid for visualization
        width, height = 800, 600
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        coordinates = np.stack([X, Y], axis=-1)
        
        # Compute textual field from all seeds
        textual_field = np.zeros((height, width))
        
        for seed in self.text_seeds:
            local_field = seed.compute_textual_field(coordinates, self.context, time)
            textual_field += local_field
        
        # Normalize field
        if textual_field.max() > textual_field.min():
            textual_field = (textual_field - textual_field.min()) / (textual_field.max() - textual_field.min())
        
        # Create color mapping
        hue = textual_field
        saturation = np.full_like(textual_field, 0.7 + 0.3 * np.sin(time * 2))
        value = 0.3 + 0.7 * textual_field.astype(float)
        
        # Convert to RGB
        hsv_image = np.stack([hue, saturation, value], axis=-1)
        rgb_image = hsv_to_rgb(hsv_image)
        
        return rgb_image, textual_field
    
    def render_text_analysis(self, text, time=0.0):
        """Render text with visual analysis"""
        # Generate visualization
        rgb_image, textual_field = self.generate_text_visualization(text, time)
        
        # Create figure with text and visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='black')
        
        # Left: Text display
        ax1.text(0.1, 0.9, 'Generated Text:', transform=ax1.transAxes, 
                color='white', fontsize=16, fontweight='bold')
        ax1.text(0.1, 0.8, text[:200] + '...' if len(text) > 200 else text, 
                transform=ax1.transAxes, color='white', fontsize=12, 
                verticalalignment='top', wrap=True)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title(f'CE1 Text Generation - {self.text_domain.title()}', 
                     color='white', fontsize=14)
        
        # Right: Textual field visualization
        im = ax2.imshow(rgb_image, extent=(-2, 2, -2, 2))
        ax2.set_title('Textual Field Visualization', color='white', fontsize=14)
        ax2.axis('off')
        
        # Add seed locations
        for seed in self.text_seeds:
            x, y = seed.position
            ax2.scatter(x, y, c='red', s=50, alpha=0.8, edgecolors='white')
        
        plt.tight_layout()
        plt.show()
        
        return rgb_image

def main():
    """Generate text using the CE1 seed computer architecture"""
    print("🚀 CE1 Text Generation System")
    print("=" * 50)
    print("Each seed is a computer that generates text through distributed computation")
    
    # Test different text domains
    text_domains = ['poetry', 'prose', 'technical', 'creative']
    
    for domain in text_domains:
        print(f"\n📝 Generating {domain} text...")
        
        # Create text generator
        text_gen = CE1TextGenerator(num_seeds=24, text_domain=domain)
        
        # Generate text
        generated_text = text_gen.generate_text(target_length=150, time=0.0)
        
        print(f"\nGenerated {domain} text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Create visualization
        print(f"\n🎨 Creating {domain} text visualization...")
        rgb_image = text_gen.render_text_analysis(generated_text, time=0.0)
        
        # Save visualization
        filename = f'.in/ce1_text_{domain}.png'
        plt.imsave(filename, rgb_image)
        print(f"Saved visualization: {filename}")
        
        # Save text
        text_filename = f'.in/ce1_text_{domain}.txt'
        with open(text_filename, 'w') as f:
            f.write(generated_text)
        print(f"Saved text: {text_filename}")
    
    print("\n✨ Text generation complete!")
    print("Each text emerged from the collective computation of distributed seed computers.")
    print("The seeds aren't just generating patterns—they're writing, thinking, and creating language!")

if __name__ == "__main__":
    main()
