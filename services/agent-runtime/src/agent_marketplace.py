"""
Agent Marketplace - Dynamic Agent Loading and Registry
"""
import json
import importlib.util
import inspect
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import structlog
from google.cloud import storage, firestore
from pathlib import Path
import asyncio
import hashlib

from models import AgentRequest, AgentResponse
from agents.base import BaseAgent
from config import Settings

logger = structlog.get_logger()

class AgentMarketplace:
    """Manages dynamic agent loading and marketplace registry"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage_client = storage.Client(project=settings.PROJECT_ID)
        self.firestore_client = firestore.Client(project=settings.PROJECT_ID)
        self.marketplace_bucket = self.storage_client.bucket(settings.POML_BUCKET)
        self.agents_collection = self.firestore_client.collection('agent_marketplace')
        self.loaded_agents: Dict[str, BaseAgent] = {}
        self.agent_metadata: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize marketplace and load registered agents"""
        try:
            # Load agent registry from Firestore
            await self.refresh_registry()
            
            # Load enabled agents
            await self.load_enabled_agents()
            
            logger.info(f"Agent Marketplace initialized with {len(self.loaded_agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Marketplace", error=str(e))
            raise
    
    async def refresh_registry(self):
        """Refresh agent registry from Firestore"""
        try:
            docs = self.agents_collection.where('status', '==', 'active').stream()
            
            self.agent_metadata = {}
            for doc in docs:
                agent_data = doc.to_dict()
                self.agent_metadata[doc.id] = {
                    **agent_data,
                    'id': doc.id,
                    'last_updated': agent_data.get('last_updated', datetime.utcnow())
                }
                
            logger.info(f"Loaded {len(self.agent_metadata)} agents from registry")
            
        except Exception as e:
            logger.error(f"Failed to refresh registry", error=str(e))
            raise
    
    async def register_agent(self, agent_config: Dict[str, Any]) -> str:
        """Register a new agent in the marketplace"""
        try:
            # Validate agent configuration
            required_fields = ['name', 'version', 'description', 'category', 'author']
            for field in required_fields:
                if field not in agent_config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate agent ID
            agent_id = f"{agent_config['name']}_{agent_config['version']}".lower().replace(' ', '_')
            
            # Add metadata
            agent_config.update({
                'id': agent_id,
                'status': 'active',
                'registered_at': datetime.utcnow(),
                'last_updated': datetime.utcnow(),
                'downloads': 0,
                'rating': 0.0,
                'reviews': []
            })
            
            # Store in Firestore
            self.agents_collection.document(agent_id).set(agent_config)
            
            # Upload agent code if provided
            if 'code' in agent_config:
                await self.upload_agent_code(agent_id, agent_config['code'])
            
            # Upload POML template if provided
            if 'poml_template' in agent_config:
                await self.upload_poml_template(agent_id, agent_config['poml_template'])
            
            logger.info(f"Registered agent: {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to register agent", error=str(e))
            raise
    
    async def upload_agent_code(self, agent_id: str, code: str):
        """Upload agent code to Cloud Storage"""
        try:
            blob_name = f"agents/{agent_id}/agent.py"
            blob = self.marketplace_bucket.blob(blob_name)
            blob.upload_from_string(code)
            
            # Update metadata with code location
            self.agents_collection.document(agent_id).update({
                'code_location': f"gs://{self.settings.POML_BUCKET}/{blob_name}",
                'code_hash': hashlib.sha256(code.encode()).hexdigest()
            })
            
        except Exception as e:
            logger.error(f"Failed to upload agent code", agent_id=agent_id, error=str(e))
            raise
    
    async def upload_poml_template(self, agent_id: str, template: str):
        """Upload POML template to Cloud Storage"""
        try:
            blob_name = f"agents/{agent_id}/template.poml"
            blob = self.marketplace_bucket.blob(blob_name)
            blob.upload_from_string(template)
            
            # Update metadata with template location
            self.agents_collection.document(agent_id).update({
                'poml_template_location': f"gs://{self.settings.POML_BUCKET}/{blob_name}",
                'template_hash': hashlib.sha256(template.encode()).hexdigest()
            })
            
        except Exception as e:
            logger.error(f"Failed to upload POML template", agent_id=agent_id, error=str(e))
            raise
    
    async def load_agent(self, agent_id: str) -> BaseAgent:
        """Dynamically load an agent from the marketplace"""
        try:
            # Check if already loaded
            if agent_id in self.loaded_agents:
                return self.loaded_agents[agent_id]
            
            # Get agent metadata
            metadata = self.agent_metadata.get(agent_id)
            if not metadata:
                # Try to fetch from Firestore
                doc = self.agents_collection.document(agent_id).get()
                if not doc.exists:
                    raise ValueError(f"Agent {agent_id} not found in marketplace")
                metadata = doc.to_dict()
            
            # Download agent code
            code_location = metadata.get('code_location')
            if not code_location:
                raise ValueError(f"No code location for agent {agent_id}")
            
            # Download from Cloud Storage
            blob_name = code_location.replace(f"gs://{self.settings.POML_BUCKET}/", "")
            blob = self.marketplace_bucket.blob(blob_name)
            code = blob.download_as_text()
            
            # Create temporary module and load
            agent_class = await self._load_agent_from_code(agent_id, code)
            
            # Instantiate agent
            agent_instance = agent_class(self.settings)
            await agent_instance.initialize()
            
            # Load POML template if available
            if metadata.get('poml_template_location'):
                template = await self.load_poml_template(agent_id)
                agent_instance.poml_template = template
            
            # Cache the loaded agent
            self.loaded_agents[agent_id] = agent_instance
            
            # Update download count
            self.agents_collection.document(agent_id).update({
                'downloads': firestore.Increment(1),
                'last_accessed': datetime.utcnow()
            })
            
            logger.info(f"Loaded agent: {agent_id}")
            return agent_instance
            
        except Exception as e:
            logger.error(f"Failed to load agent", agent_id=agent_id, error=str(e))
            raise
    
    async def _load_agent_from_code(self, agent_id: str, code: str) -> Type[BaseAgent]:
        """Load agent class from code string"""
        try:
            # Create a module spec
            spec = importlib.util.spec_from_loader(
                f"dynamic_agent_{agent_id}",
                loader=None
            )
            
            # Create module
            module = importlib.util.module_from_spec(spec)
            
            # Execute code in module namespace
            exec(code, module.__dict__)
            
            # Find the agent class (should inherit from BaseAgent)
            agent_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj != BaseAgent:
                    agent_class = obj
                    break
            
            if not agent_class:
                raise ValueError(f"No valid agent class found in code for {agent_id}")
            
            return agent_class
            
        except Exception as e:
            logger.error(f"Failed to load agent from code", agent_id=agent_id, error=str(e))
            raise
    
    async def load_poml_template(self, agent_id: str) -> str:
        """Load POML template for an agent"""
        try:
            metadata = self.agent_metadata.get(agent_id)
            if not metadata or not metadata.get('poml_template_location'):
                return ""
            
            blob_name = metadata['poml_template_location'].replace(f"gs://{self.settings.POML_BUCKET}/", "")
            blob = self.marketplace_bucket.blob(blob_name)
            return blob.download_as_text()
            
        except Exception as e:
            logger.error(f"Failed to load POML template", agent_id=agent_id, error=str(e))
            return ""
    
    async def load_enabled_agents(self):
        """Load all enabled agents"""
        try:
            enabled_agents = [
                agent_id for agent_id, metadata in self.agent_metadata.items()
                if metadata.get('auto_load', False) and metadata.get('status') == 'active'
            ]
            
            tasks = [self.load_agent(agent_id) for agent_id in enabled_agents]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to load enabled agents", error=str(e))
    
    async def search_agents(self, query: Dict[str, Any]) -> List[Dict]:
        """Search for agents in the marketplace"""
        try:
            results = []
            
            # Build query
            collection_ref = self.agents_collection
            
            if 'category' in query:
                collection_ref = collection_ref.where('category', '==', query['category'])
            
            if 'min_rating' in query:
                collection_ref = collection_ref.where('rating', '>=', query['min_rating'])
            
            if 'author' in query:
                collection_ref = collection_ref.where('author', '==', query['author'])
            
            # Execute query
            docs = collection_ref.stream()
            
            for doc in docs:
                agent_data = doc.to_dict()
                
                # Text search in name and description
                if 'search_text' in query:
                    search_text = query['search_text'].lower()
                    if (search_text not in agent_data.get('name', '').lower() and
                        search_text not in agent_data.get('description', '').lower()):
                        continue
                
                results.append(agent_data)
            
            # Sort by rating or downloads
            sort_by = query.get('sort_by', 'rating')
            results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            
            return results[:query.get('limit', 50)]
            
        except Exception as e:
            logger.error(f"Failed to search agents", error=str(e))
            return []
    
    async def rate_agent(self, agent_id: str, rating: float, review: Optional[str] = None):
        """Rate an agent in the marketplace"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent_data = doc.to_dict()
            reviews = agent_data.get('reviews', [])
            
            # Add review
            review_data = {
                'rating': rating,
                'review': review,
                'timestamp': datetime.utcnow(),
                'user': 'anonymous'  # In production, get from auth
            }
            reviews.append(review_data)
            
            # Calculate new average rating
            avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
            
            # Update document
            doc_ref.update({
                'rating': avg_rating,
                'reviews': reviews,
                'review_count': len(reviews)
            })
            
            logger.info(f"Rated agent {agent_id}: {rating}")
            
        except Exception as e:
            logger.error(f"Failed to rate agent", agent_id=agent_id, error=str(e))
            raise
    
    def get_loaded_agents(self) -> Dict[str, BaseAgent]:
        """Get all currently loaded agents"""
        return self.loaded_agents
    
    def get_agent_metadata(self, agent_id: str) -> Optional[Dict]:
        """Get metadata for a specific agent"""
        return self.agent_metadata.get(agent_id)
    
    async def unload_agent(self, agent_id: str):
        """Unload an agent from memory"""
        if agent_id in self.loaded_agents:
            agent = self.loaded_agents[agent_id]
            await agent.cleanup()
            del self.loaded_agents[agent_id]
            logger.info(f"Unloaded agent: {agent_id}")
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]):
        """Update agent configuration"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            updates['last_updated'] = datetime.utcnow()
            doc_ref.update(updates)
            
            # Reload if already loaded
            if agent_id in self.loaded_agents:
                await self.unload_agent(agent_id)
                await self.load_agent(agent_id)
            
            logger.info(f"Updated agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to update agent", agent_id=agent_id, error=str(e))
            raise