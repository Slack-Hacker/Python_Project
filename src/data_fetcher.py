import asyncio
import aiohttp
import pandas as pd

class AsyncDataFetcher:
    def __init__(self, urls):
        self.urls = urls

    async def _fetch_url(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return {'url': url, 'data': data, 'status': 'success'}
                return {'url': url, 'data': None, 'status': f'http_{response.status}'}
        except Exception as e:
            return {'url': url, 'data': None, 'status': f'error_{str(e)}'}

    async def fetch_all(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, url) for url in self.urls]
            results = await asyncio.gather(*tasks)
            return results

def run_harvesting(urls):
    fetcher = AsyncDataFetcher(urls)
    return asyncio.run(fetcher.fetch_all())
