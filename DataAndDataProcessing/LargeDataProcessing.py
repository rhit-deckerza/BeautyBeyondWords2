import json
import pandas as pd
from spacy.matcher import PhraseMatcher
import spacy
import re
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import os

tqdm.pandas()

print("Loading product data...")
data = []
with open(r'C:\Users\zadec\Desktop\Cornell Tech\Deep Learning\Final Project\real\meta_Beauty_and_Personal_Care.jsonl\meta_Beauty_and_Personal_Care.jsonl', 'r') as f:
    for line in tqdm(f, desc="Reading JSON lines"):
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line due to error: {e}")

print("Creating products DataFrame...")
products = pd.DataFrame.from_records(data)
products = products[(products['description'].apply(lambda x: len(x) > 0)) | (products['features'].apply(lambda x: len(x) > 0))]

print("Defining ingredients and compiling regex pattern...")
ingredients = [
    "Allantoin", "Alcohol Denat", "Almond Oil", "Aloe Vera", "Alpha Hydroxy Acid", "Amala Oil", "Amino Acids",
    "Amoxicillin", "Antioxidants", "Apple Cider Vinegar", "Apricot Kernel Oil", "Arbutin", "Argan Oil",
    "Argireline", "Ascorbyl Glucoside", "Astaxanthin", "Avocado Oil", "Azelaic Acid", "Azulene", "Baobab",
    "Baking Soda", "Bakuchiol", "Bentonite Clay", "Benzoyl Peroxide", "Benzyl Alcohol", "Beta Glucan",
    "Bhringraj Oil", "Biotin", "Bio Oil", "Black Cumin Seed Oil", "Borage Seed Oil", "Butylene Glycol", "CBD Oil",
    "CBD", "Caffeine", "Calamine Lotion", "Camellia Extract", "Capric Triglyceride", "Caprylyl Glycol", "Carbomer",
    "Caviar Extract", "Carrier Oils", "Carrot", "Castor Oil", "Cephalexin", "Ceramides", "Cetearyl Alcohol",
    "Chamomile", "Charcoal", "Chebula", "Chia Seed Oil", "Citric Acid", "Cocamidopropyl-Betaine", "Cocoa Butter",
    "Coconut Oil", "Collagen", "Colloidal Oatmeal", "Cone Snail Venom", "Copper Peptides", "CoQ10",
    "Cyclopentasiloxane", "Cypress Oil", "Desitin", "Dihydroxyacetone", "Dimethicone", "Doxycycline", "Emollients",
    "Emu Oil", "Epsom Salt", "Eucalyptus Oil", "Evening Primrose Oil", "Ferulic Acid", "Fermented Oils",
    "Frangipani", "Gluconolactone", "Glycerin", "Glycolic Acid", "Goat's Milk", "Goji Berry", "Gold",
    "Grapeseed Oil", "Green Tea", "Hemp Oil", "Homosalate", "Honey", "Humectants", "Hyaluronic Acid",
    "Hydrocortisone", "Hydrogen Peroxide", "Hydroquinone", "Isododecane", "Isoparaffin", "Isopropyl Myristate",
    "Jojoba", "Kaolin", "Karanja Oil", "Kigelia Africana", "Kojic Acid", "Kukui Nut Oil", "Lactic Acid",
    "Lactobionic Acid", "Lanolin", "Lavender Oil", "Lemon Juice", "Licorice Extract", "Lysine", "Madecassoside",
    "Magnesium", "Magnesium Aluminum Silicate", "Malic Acid", "Mandelic Acid", "Manuka Honey",
    "Marshmallow Root Extract", "Marula Oil", "Meadowfoam", "Methylparaben", "Mineral Oil", "Moringa Oil",
    "Murumuru Butter", "Muslin", "Neem Oil", "Niacinamide", "Nizoral", "Oat", "Octinoxate", "Octisalate",
    "Octocrylene", "Olive Oil", "Omega Fatty Acids", "Oxybenzone", "Panthenol", "Parabens", "Peppermint Oil",
    "Petroleum Jelly", "PHA", "Phenoxyethanol", "Phytic Acid", "Phytosphingosine", "Placenta", "Plum Oil",
    "Polyglutamic Acid", "Polypeptides", "Pomegranates", "Prickly Pear Oil", "Probioitics", "Progeline",
    "Propanediol", "Propolis", "Propylene Glycol", "Propylparabens", "Purslane Extract", "Pycnogenol",
    "Quercetin", "Reishi Mushrooms", "Resveratrol", "Retin-A", "Retinaldehyde", "Retinol", "Retinyl Palmitate",
    "Rosehip Oil", "Rosemary", "Royal Jelly", "Safflower Oil", "Salicylic Acid", "Sea Buckthorn Oil", "Sea Salt",
    "Seaweed", "Sea Moss", "Shea Butter", "Silver", "Snail Mucin", "Sodium Ascorbyl Phosphate", "Sodium Deoxycholate",
    "Sodium Hyaluronate", "Sodium Hydroxide", "Sodium Lauroyl Lactylate", "Sodium Lauryl Sulfate", "Sodium Palmate",
    "Sodium PCA", "Sodium Tallowate", "Soybean Oil", "Spironolactone", "Stearic Acid", "Stearyl Alcohol",
    "Squalane", "Stem Cells", "Succinic Acid", "Sulfates", "Sulfur", "Sunflower Oil", "Synthetic Beeswax", "Talc",
    "Tamanu Oil", "Tea Tree Oil", "Tepezcohuite", "Tranexamic Acid", "Tretinoin", "Triethanolamine", "Turmeric",
    "Undecylenic Acid", "Urea 40", "Vegetable Glycerin", "Vitamin A", "Vitamin B3", "Vitamin C", "Vitamin D",
    "Vitamin E", "Vitamin F", "Vitamin K", "Volcanic Ash", "Willow Bark Extract", "Xanthan Gum", "Zinc"
]
ingredients_pattern = "|".join([re.escape(ingredient.lower()) for ingredient in ingredients])
ingredients_regex = re.compile(ingredients_pattern)
def extract_unique_ingredients_regex(description, ingredients_regex):
    """
    Extract unique ingredients dynamically from a description using regex.
    """
    matches = ingredients_regex.findall(description.lower())
    return sorted(set(matches))
print("Extracting ingredients from product descriptions...")
products['combined_description'] = products['features'].astype(str) + " " + products['description'].astype(str)
products['extracted_ingredients'] = products['combined_description'].progress_apply(
    lambda desc: extract_unique_ingredients_regex(desc, ingredients_regex)
)
products = products[products['extracted_ingredients'].apply(lambda x: len(x) > 0)]

print("Defining skin features...")
skin_features = {
    "Normal Skin": ["normal", "balanced", "healthy", "clear", "untroubled", "even"],
    "Oily Skin": ["oily", "greasy", "shiny", "excess sebum", "slick", "glossy"],
    "Combination Skin": ["combination", "mixed", "dry and oily", "dual type", "patchy oily"],
    "Sensitive Skin": ["sensitive", "irritation", "reactive", "allergic", "fragile", "delicate", "easily irritated", "prone to redness"],
    "Acne": ["acne", "pimple", "blemish", "breakout", "zits", "cystic acne", "spots", "acne-prone", "comedones"],
    "Hydration": ["hydrating", "moisture", "moisturizing", "replenish", "quenched", "hydrated", "plumping", "water retention", "moist"],
    "Pores": ["pores", "enlarged pores", "pore size", "clogged pores", "minimize pores", "visible pores", "pore congestion"],
    "Fine Lines and Wrinkles": ["wrinkles", "fine lines", "aging", "anti-aging", "crow's feet", "expression lines", "laugh lines", "age lines", "crinkles"],
    "Sagging": ["sagging", "loose", "loss of firmness", "drooping", "lack of elasticity", "laxity", "lifting", "skin slackening", "gravity-prone"],
    "Dark Spots": ["dark spots", "hyperpigmentation", "discoloration", "sun spots", "age spots", "melasma", "uneven tone", "brown patches", "pigmented spots"],
    "Redness": ["redness", "red patches", "inflammation", "rosacea", "flushed", "red blotches", "irritated skin", "blushing", "hyperemia"],
    "Uneven Texture": ["uneven texture", "rough", "bumpy", "textured", "dull", "grainy", "coarse", "patchy", "irregular texture"],
    "Dark Circles": ["dark circles", "under-eye", "eye bags", "puffy eyes", "tired eyes", "under-eye discoloration", "shadow", "hollows", "dark under-eyes"]
}

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

def extract_unique_skin_features(text, nlp, skin_features):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for feature, related_phrases in skin_features.items():
        patterns = [nlp.make_doc(phrase) for phrase in [feature] + related_phrases]
        matcher.add(feature, patterns)
    doc = nlp(text)
    matches = matcher(doc)
    unique_features = {nlp.vocab.strings[match_id].lower() for match_id, _, _ in matches}
    return sorted(unique_features)

checkpoint_file = 'checkpoint.txt'
temp_output_file = 'temp_preprocessed.csv'

# Load checkpoint if it exists
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        last_processed_chunk = int(f.read().strip())
else:
    last_processed_chunk = -1

print("Processing reviews in chunks...")
chunk_size = 1000
with open(r'C:\Users\zadec\Desktop\Cornell Tech\Deep Learning\Final Project\real\meta_Beauty_and_Personal_Care.jsonl\meta_Beauty_and_Personal_Care.jsonl', 'r') as f:
    total_lines = sum(1 for _ in f)
    total_chunks = (total_lines // chunk_size) + (1 if total_lines % chunk_size != 0 else 0)
    f.seek(0)  # Reset file pointer to the beginning

    total_attributes = 0
    total_reviews = 0
    total_ingredients = 0
    total_descriptions = 0

    for chunk_num, chunk in enumerate(tqdm(pd.read_json(f, lines=True, chunksize=chunk_size), desc="Processing chunks")):
        if chunk_num <= last_processed_chunk:
            continue  # Skip already processed chunks

        print(f"\nProcessing chunk {chunk_num + 1} of {total_chunks}...")
        reviews = chunk
        
        print("Merging products and reviews...")
        reviews_match = products[['parent_asin', 'combined_description', 'title', 'extracted_ingredients']].merge(
            reviews, on='parent_asin', how='inner'
        )
        
        print("Extracting skin features...")
        reviews_match['skin_features'] = reviews_match['combined_description'].progress_apply(
            lambda review: extract_unique_skin_features(str(review), nlp, skin_features)
        )
        
        reviews_match = reviews_match[reviews_match['skin_features'].apply(lambda x: len(x) > 0)]
        
        # Calculate running average of attributes per review
        total_attributes += reviews_match['skin_features'].apply(len).sum()
        total_reviews += len(reviews_match)
        running_average_attributes = total_attributes / total_reviews if total_reviews > 0 else 0
        print(f"Running average number of attributes per review: {running_average_attributes:.2f}")
        
        # Calculate running average of ingredients per description
        total_ingredients += reviews_match['extracted_ingredients'].apply(len).sum()
        total_descriptions += len(reviews_match)
        running_average_ingredients = total_ingredients / total_descriptions if total_descriptions > 0 else 0
        print(f"Running average number of ingredients per description: {running_average_ingredients:.2f}")
        
        print("Grouping and encoding features...")
        reviews_match['extracted_ingredients'] = reviews_match['extracted_ingredients'].apply(tuple)
        grouped_reviews = reviews_match.groupby(['parent_asin', 'extracted_ingredients'])['skin_features'].apply(lambda x: [attr for sublist in x for attr in sublist])
        grouped_reviews = grouped_reviews.apply(lambda x: list(set(x)))
        grouped_reviews_df = grouped_reviews.reset_index()
        grouped_reviews_df = grouped_reviews_df.merge(products[['parent_asin', 'title']], on='parent_asin', how='left')
        
        grouped_reviews_df['ingredients_list'] = grouped_reviews_df['extracted_ingredients'].apply(lambda x: ', '.join(x))
        
        print("Encoding skin features...")
        encoded_skin_features = [set(features) for features in grouped_reviews_df['skin_features']]
        mlb_features = MultiLabelBinarizer()
        features_encoded = mlb_features.fit_transform(encoded_skin_features)
        features_df = pd.DataFrame(features_encoded, columns=mlb_features.classes_)
        
        print("Creating final DataFrame...")
        modeling_df = pd.concat([grouped_reviews_df['parent_asin'], grouped_reviews_df['title'], grouped_reviews_df['ingredients_list'], features_df], axis=1)
        
        # Save intermediate results
        if os.path.exists(temp_output_file):
            temp_df = pd.read_csv(temp_output_file)
            temp_df = pd.concat([temp_df, modeling_df])
        else:
            temp_df = modeling_df

        temp_df.to_csv(temp_output_file, index=False)

        # Update checkpoint
        with open(checkpoint_file, 'w') as f:
            f.write(str(chunk_num))

# Finalize the output
if os.path.exists(temp_output_file):
    try:
        existing_df = pd.read_csv('preprocessed.csv')
        combined_df = pd.concat([existing_df, temp_df])
        
        # Group by 'parent_asin' and apply the logic for combining entries
        combined_df = combined_df.groupby('parent_asin').agg({
            'title': 'last',  # Replace title with the last occurrence
            'ingredients_list': lambda x: ', '.join(set(', '.join(x).split(', '))),  # Combine and deduplicate ingredients
            **{col: 'max' for col in features_df.columns}  # Apply OR operation for binary features
        }).reset_index()
    except FileNotFoundError:
        combined_df = temp_df

    combined_df.to_csv('preprocessed.csv', index=False)

    # Clean up temporary files
    os.remove(temp_output_file)
    os.remove(checkpoint_file)

print("\nProcessing complete!")