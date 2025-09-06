import pytest
import httpx

BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_api_is_running():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/docs")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_search_with_batch():
    async with httpx.AsyncClient() as client:
        # Test with batch 1
        response = await client.post(
            f"{BASE_URL}/api/v1/keyframe/search?batch=1",
            json={"query": "test", "top_k": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # Test with batch 2
        response = await client.post(
            f"{BASE_URL}/api/v1/keyframe/search?batch=2",
            json={"query": "test", "top_k": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

@pytest.mark.asyncio
async def test_search_exclude_group():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/keyframe/search/exclude-groups?batch=1",
            json={"query": "test", "top_k": 10, "exclude_groups": [1]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Further assertions would require a known dataset
        # For now, we just check that the request is successful
