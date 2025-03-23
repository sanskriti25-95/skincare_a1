import json
import asyncio
from product_api import ProductAPI

async def create_recommendations():
    api = ProductAPI()
    
    recommendations = {
        "oily": {
            "routine": "Morning:\n1. Gentle foaming cleanser\n2. Alcohol-free toner\n3. Light gel moisturizer\n4. Sunscreen (SPF 50+)\n\nEvening:\n1. Double cleanse\n2. BHA/Salicylic acid treatment\n3. Light moisturizer",
            "products": {
                "cleansers": await api.get_products_by_skin_type('oily'),
                "moisturizers": await api.get_products_by_skin_type('oily'),
                "treatments": await api.get_products_by_skin_type('oily'),
                "sunscreens": await api.get_products_by_skin_type('oily')
            },
            "home_remedies": [
                "Multani mitti (Fuller's earth) face pack twice weekly",
                "Neem and tulsi face pack",
                "Rose water as natural toner",
                "Aloe vera gel for oil control"
            ]
        },
        "dry": {
            "routine": "Morning:\n1. Cream cleanser\n2. Hydrating toner\n3. Rich moisturizer\n4. Sunscreen (SPF 30+)\n\nEvening:\n1. Oil-based cleanser\n2. Hydrating serum\n3. Night cream\n4. Face oil",
            "products": await api.get_products_by_skin_type('dry'),
            "home_remedies": [
                "Honey and malai (cream) face mask",
                "Almond oil massage",
                "Mashed banana and honey pack",
                "Besan and curd pack"
            ]
        },
        "normal": {
            "routine": "Morning:\n1. Gentle cleanser\n2. Toner\n3. Light moisturizer\n4. Sunscreen\n\nEvening:\n1. Cleanser\n2. Treatment serum\n3. Night moisturizer",
            "products": await api.get_products_by_skin_type('normal'),
            "home_remedies": [
                "Rosewater and glycerin toner",
                "Weekly besan (gram flour) scrub",
                "Cucumber and aloe vera gel pack",
                "Papaya pulp mask for glow"
            ]
        }
    }

    with open('product_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=4)

    print("Created product_recommendations.json with real-time product data")

if __name__ == "__main__":
    asyncio.run(create_recommendations()) 