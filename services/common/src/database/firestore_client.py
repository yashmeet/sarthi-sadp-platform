"""
Production Firestore Client with Connection Pooling and Error Handling
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import structlog
import asyncio
from dataclasses import asdict

logger = structlog.get_logger()

class FirestoreClient:
    """
    Production-ready Firestore client with connection pooling and error handling
    """
    
    def __init__(self, project_id: str = None, database_id: str = "(default)"):
        self.project_id = project_id
        self.database_id = database_id
        self.client = None
        self.initialized = False
        self.retry_attempts = 3
        self.metrics = {
            "reads": 0,
            "writes": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize Firestore client"""
        try:
            self.client = firestore.Client(
                project=self.project_id,
                database=self.database_id
            )
            
            # Test connection
            await self._test_connection()
            
            self.initialized = True
            logger.info("Firestore client initialized", 
                       project_id=self.project_id,
                       database_id=self.database_id)
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Firestore client", error=str(e))
            return False
    
    async def _test_connection(self):
        """Test Firestore connection"""
        try:
            # Try to read from a test collection
            test_ref = self.client.collection('_health_check')
            test_ref.limit(1).get()
        except Exception as e:
            logger.error("Firestore connection test failed", error=str(e))
            raise
    
    async def create_document(
        self, 
        collection_path: str, 
        document_data: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """Create a new document"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Add timestamps
            document_data['created_at'] = datetime.now(timezone.utc)
            document_data['updated_at'] = datetime.now(timezone.utc)
            
            collection_ref = self.client.collection(collection_path)
            
            if document_id:
                doc_ref = collection_ref.document(document_id)
                doc_ref.set(document_data)
                created_id = document_id
            else:
                doc_ref = collection_ref.add(document_data)[1]
                created_id = doc_ref.id
            
            self.metrics["writes"] += 1
            logger.info("Document created",
                       collection=collection_path,
                       document_id=created_id)
            
            return created_id
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to create document",
                        collection=collection_path,
                        error=str(e))
            raise
    
    async def get_document(
        self, 
        collection_path: str, 
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        if not self.initialized:
            await self.initialize()
        
        try:
            doc_ref = self.client.collection(collection_path).document(document_id)
            doc = doc_ref.get()
            
            self.metrics["reads"] += 1
            
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                logger.debug("Document retrieved",
                           collection=collection_path,
                           document_id=document_id)
                return data
            else:
                logger.debug("Document not found",
                           collection=collection_path,
                           document_id=document_id)
                return None
                
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to get document",
                        collection=collection_path,
                        document_id=document_id,
                        error=str(e))
            raise
    
    async def update_document(
        self,
        collection_path: str,
        document_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """Update a document"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Add update timestamp
            updates['updated_at'] = datetime.now(timezone.utc)
            
            doc_ref = self.client.collection(collection_path).document(document_id)
            
            if merge:
                doc_ref.set(updates, merge=True)
            else:
                doc_ref.update(updates)
            
            self.metrics["writes"] += 1
            logger.info("Document updated",
                       collection=collection_path,
                       document_id=document_id)
            
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to update document",
                        collection=collection_path,
                        document_id=document_id,
                        error=str(e))
            raise
    
    async def delete_document(
        self,
        collection_path: str,
        document_id: str
    ) -> bool:
        """Delete a document"""
        if not self.initialized:
            await self.initialize()
        
        try:
            doc_ref = self.client.collection(collection_path).document(document_id)
            doc_ref.delete()
            
            self.metrics["deletes"] += 1
            logger.info("Document deleted",
                       collection=collection_path,
                       document_id=document_id)
            
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to delete document",
                        collection=collection_path,
                        document_id=document_id,
                        error=str(e))
            raise
    
    async def query_documents(
        self,
        collection_path: str,
        filters: List[Dict[str, Any]] = None,
        order_by: str = None,
        limit: int = None,
        start_after: str = None
    ) -> List[Dict[str, Any]]:
        """Query documents with filters"""
        if not self.initialized:
            await self.initialize()
        
        try:
            query = self.client.collection(collection_path)
            
            # Apply filters
            if filters:
                for filter_dict in filters:
                    field = filter_dict['field']
                    operator = filter_dict['operator']
                    value = filter_dict['value']
                    query = query.where(filter=FieldFilter(field, operator, value))
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    query = query.order_by(order_by[1:], direction=firestore.Query.DESCENDING)
                else:
                    query = query.order_by(order_by, direction=firestore.Query.ASCENDING)
            
            # Apply pagination
            if start_after:
                start_doc = self.client.collection(collection_path).document(start_after).get()
                query = query.start_after(start_doc)
            
            if limit:
                query = query.limit(limit)
            
            # Execute query
            docs = query.stream()
            results = []
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)
            
            self.metrics["reads"] += len(results)
            logger.debug("Query executed",
                        collection=collection_path,
                        results_count=len(results))
            
            return results
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to query documents",
                        collection=collection_path,
                        error=str(e))
            raise
    
    async def batch_write(
        self,
        operations: List[Dict[str, Any]]
    ) -> bool:
        """Execute batch write operations"""
        if not self.initialized:
            await self.initialize()
        
        try:
            batch = self.client.batch()
            
            for op in operations:
                op_type = op['type']  # 'create', 'update', 'delete'
                collection_path = op['collection']
                document_id = op['document_id']
                
                doc_ref = self.client.collection(collection_path).document(document_id)
                
                if op_type == 'create':
                    data = op['data']
                    data['created_at'] = datetime.now(timezone.utc)
                    data['updated_at'] = datetime.now(timezone.utc)
                    batch.set(doc_ref, data)
                    
                elif op_type == 'update':
                    data = op['data']
                    data['updated_at'] = datetime.now(timezone.utc)
                    batch.update(doc_ref, data)
                    
                elif op_type == 'delete':
                    batch.delete(doc_ref)
            
            # Commit batch
            batch.commit()
            
            self.metrics["writes"] += len(operations)
            logger.info("Batch operation completed", operations_count=len(operations))
            
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error("Failed to execute batch operation", error=str(e))
            raise
    
    async def listen_to_collection(
        self,
        collection_path: str,
        callback,
        filters: List[Dict[str, Any]] = None
    ):
        """Listen to real-time updates on a collection"""
        if not self.initialized:
            await self.initialize()
        
        try:
            query = self.client.collection(collection_path)
            
            # Apply filters
            if filters:
                for filter_dict in filters:
                    field = filter_dict['field']
                    operator = filter_dict['operator']
                    value = filter_dict['value']
                    query = query.where(filter=FieldFilter(field, operator, value))
            
            # Set up listener
            def on_snapshot(doc_snapshot, changes, read_time):
                for change in changes:
                    if change.type.name == 'ADDED':
                        callback('added', change.document.to_dict())
                    elif change.type.name == 'MODIFIED':
                        callback('modified', change.document.to_dict())
                    elif change.type.name == 'REMOVED':
                        callback('removed', change.document.to_dict())
            
            # Start listening
            query.on_snapshot(on_snapshot)
            
            logger.info("Real-time listener started", collection=collection_path)
            
        except Exception as e:
            logger.error("Failed to set up listener", 
                        collection=collection_path,
                        error=str(e))
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get database metrics"""
        total_operations = self.metrics["reads"] + self.metrics["writes"] + self.metrics["deletes"]
        
        return {
            **self.metrics,
            "total_operations": total_operations,
            "error_rate": self.metrics["errors"] / max(total_operations, 1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Firestore"""
        try:
            await self._test_connection()
            return {
                "status": "healthy",
                "initialized": self.initialized,
                "metrics": self.get_metrics()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.initialized
            }

# Global Firestore client instance
firestore_client = FirestoreClient()