# ==============================================================================
# 🔬 VOCABULARY ANALYSIS: Identify Extra Medical Tokens
# ==============================================================================
# This cell identifies the vocabulary differences between Meditron-70B and Llama-2-7B
# Run this AFTER HuggingFace authentication to see which tokens will be added

from transformers import AutoTokenizer

print("="*80)
print("🔬 VOCABULARY ANALYSIS: Finding Extra Medical Tokens")
print("="*80)

# Load both tokenizers
print("\n📥 Loading tokenizers...")
teacher_tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70b")
student_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Get vocabularies
teacher_vocab = teacher_tokenizer.get_vocab()
student_vocab = student_tokenizer.get_vocab()

print(f"\n📊 Vocabulary Sizes:")
print(f"  Teacher (Meditron-70B): {len(teacher_vocab):,}")
print(f"  Student (Llama-2-7B):   {len(student_vocab):,}")
print(f"  Difference:             {len(teacher_vocab) - len(student_vocab)}")

# Find extra tokens
teacher_tokens = set(teacher_vocab.keys())
student_tokens = set(student_vocab.keys())
extra_tokens = sorted(list(teacher_tokens - student_tokens))

if extra_tokens:
    print(f"\n🔍 THE {len(extra_tokens)} EXTRA MEDITRON MEDICAL TOKENS:")
    print("="*80)
    for i, token in enumerate(extra_tokens, 1):
        token_id = teacher_vocab[token]
        try:
            decoded = teacher_tokenizer.decode([token_id]).strip()
            if decoded:
                print(f"{i:2d}. Token ID {token_id:>5d}: '{token}' → \"{decoded}\"")
            else:
                print(f"{i:2d}. Token ID {token_id:>5d}: '{token}' (special/control token)")
        except:
            print(f"{i:2d}. Token ID {token_id:>5d}: '{token}'")
    
    print("="*80)
    print(f"\n💡 WHAT THIS MEANS:")
    print(f"  📌 These are medical domain-specific tokens added to Meditron-70B")
    print(f"  📌 Likely includes: drug names, medical abbreviations, anatomical terms")
    print(f"  📌 Represents {len(extra_tokens)/len(teacher_vocab)*100:.3f}% of teacher vocabulary")
    
    print(f"\n✅ AUTOMATIC HANDLING IN TRAINING:")
    print(f"  🔧 For LOGIT-KD, SPIN, and other logit-based methods:")
    print(f"     → Training script will check --align_vocabularies flag")
    print(f"     → If provided: Student vocab EXPANDED with these {len(extra_tokens)} tokens")
    print(f"     → Student embeddings RESIZED automatically")
    print(f"     → New embeddings initialized from mean of existing embeddings")
    print(f"     → Enables full knowledge transfer via soft labels")
    
    print(f"\n  📝 For SFT (text generation):")
    print(f"     → NO alignment needed (--align_vocabularies not required)")
    print(f"     → Extra tokens represented as multi-token sequences automatically")
    print(f"     → Example: 'mg/dL' (1 teacher token) → ['mg', '/', 'd', 'L'] (4 student tokens)")
else:
    print("\n✅ Vocabularies already match - no alignment needed!")

print("="*80)
