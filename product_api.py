import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List
import aiohttp
import asyncio
from datetime import datetime

class ProductAPI:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Cache results for 24 hours to avoid too many API calls
        self.cache = {}
        self.cache_duration = 86400  # 24 hours in seconds

    async def fetch_nykaa_products(self, search_term: str) -> List[Dict]:
        """Fetch products from Nykaa"""
        cache_key = f"nykaa_{search_term}"
        if cache_key in self.cache:
            if (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds() < self.cache_duration:
                return self.cache[cache_key]['data']

        url = f"https://www.nykaa.com/search/result/?q={search_term}&root=search&searchType=Manual"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    products = []
                    
                    for product in soup.find_all('div', class_='product-list'):
                        try:
                            name = product.find('div', class_='product-name').text.strip()
                            price = product.find('span', class_='price').text.strip()
                            price = int(price.replace('₹', '').replace(',', '').strip())
                            rating = product.find('span', class_='rating').text.strip()
                            
                            products.append({
                                'name': name,
                                'price': price,
                                'rating': rating,
                                'source': 'Nykaa'
                            })
                        except:
                            continue
                    
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': products
                    }
                    return products
        return []

    async def fetch_amazon_products(self, search_term: str) -> List[Dict]:
        """Fetch products from Amazon"""
        cache_key = f"amazon_{search_term}"
        if cache_key in self.cache:
            if (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds() < self.cache_duration:
                return self.cache[cache_key]['data']

        url = f"https://www.amazon.in/s?k={search_term}+skincare"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    products = []
                    
                    for product in soup.find_all('div', {'data-component-type': 's-search-result'}):
                        try:
                            name = product.find('span', class_='a-text-normal').text.strip()
                            price = product.find('span', class_='a-price-whole').text.strip()
                            price = int(price.replace(',', '').strip())
                            rating = product.find('span', class_='a-icon-alt').text.strip()
                            
                            products.append({
                                'name': name,
                                'price': price,
                                'rating': rating,
                                'source': 'Amazon'
                            })
                        except:
                            continue
                    
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': products
                    }
                    return products
        return []

    async def get_products_by_skin_type(self, skin_type: str) -> Dict:
        """Get products based on skin type"""
        search_terms = {
            'oily': [
                'oil control face wash',
                'mattifying moisturizer',
                'salicylic acid',
                'niacinamide serum'
            ],
            'dry': [
                'hydrating face wash',
                'moisturizing cream',
                'hyaluronic acid serum',
                'face oil'
            ],
            'normal': [
                'gentle face wash',
                'lightweight moisturizer',
                'vitamin c serum',
                'sunscreen'
            ]
        }

        all_products = []
        tasks = []
        
        for term in search_terms[skin_type]:
            tasks.append(self.fetch_nykaa_products(term))
            tasks.append(self.fetch_amazon_products(term))
        
        results = await asyncio.gather(*tasks)
        for result in results:
            all_products.extend(result)

        # Sort by rating and price
        all_products.sort(key=lambda x: (float(x.get('rating', 0)), -x['price']), reverse=True)
        
        # Group by product type
        categorized_products = {
            'cleansers': [],
            'moisturizers': [],
            'treatments': [],
            'sunscreens': []
        }

        for product in all_products:
            name = product['name'].lower()
            if 'face wash' in name or 'cleanser' in name:
                categorized_products['cleansers'].append(product)
            elif 'moistur' in name or 'cream' in name:
                categorized_products['moisturizers'].append(product)
            elif 'serum' in name or 'treatment' in name:
                categorized_products['treatments'].append(product)
            elif 'sunscreen' in name or 'spf' in name:
                categorized_products['sunscreens'].append(product)

        # Take top 3 from each category
        for category in categorized_products:
            categorized_products[category] = categorized_products[category][:3]

        return categorized_products 