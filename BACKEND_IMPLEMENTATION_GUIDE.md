# Backend Implementation Guide for Plan Upvote/Downvote API Endpoints

## Overview

This guide provides instructions for implementing the backend Django REST Framework API endpoints for plan upvoting/downvoting functionality. The SDK client methods have been implemented in this repository and are ready to consume these endpoints once they are created.

## Task Summary

Add dedicated upvote/downvote API endpoints for Plans to provide clearer semantics and better API design for the User-Led Learning (ULL) feature.

## Backend Implementation Requirements

### 1. Database Model

The Plan model should have a `liked` boolean field:
- `liked=True` indicates an upvoted plan
- `liked=False` indicates a downvoted plan
- Default should be `False` or `null`

### 2. API Endpoints to Implement

Create the following endpoints in `/backend/project/plans/views.py`:

#### POST /api/v0/plans/{id}/upvote/
- **Method**: POST
- **Description**: Upvote a plan (sets `liked=True`)
- **Permissions**: OrgOnlyPermissions (only org members can upvote)
- **Response**: 200 OK on success
- **Side Effects**: Triggers embedding creation via model signals

#### POST /api/v0/plans/{id}/downvote/
- **Method**: POST
- **Description**: Downvote a plan (sets `liked=False`)
- **Permissions**: OrgOnlyPermissions (only org members can downvote)
- **Response**: 200 OK on success
- **Side Effects**: Triggers embedding deletion via model signals

### 3. Django ViewSet Implementation

Add these action methods to the `PlanViewSet` class:

```python
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status

class PlanViewSet(viewsets.ModelViewSet):
    permission_classes = [OrgOnlyPermissions]

    @swagger_auto_schema(
        operation_description="Upvote a plan. Sets the plan's liked field to True, "
                            "which triggers embedding creation for similarity search.",
        responses={
            200: "Plan upvoted successfully",
            404: "Plan not found",
            403: "Permission denied"
        }
    )
    @action(detail=True, methods=['post'])
    def upvote(self, request, pk=None):
        """Upvote a plan."""
        plan = self.get_object()
        plan.liked = True
        plan.save()
        # The model signals will handle embedding creation
        return Response({'status': 'success'}, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="Downvote a plan. Sets the plan's liked field to False, "
                            "which triggers embedding deletion.",
        responses={
            200: "Plan downvoted successfully",
            404: "Plan not found",
            403: "Permission denied"
        }
    )
    @action(detail=True, methods=['post'])
    def downvote(self, request, pk=None):
        """Downvote a plan."""
        plan = self.get_object()
        plan.liked = False
        plan.save()
        # The model signals will handle embedding deletion
        return Response({'status': 'success'}, status=status.HTTP_200_OK)
```

### 4. Model Signals

Ensure the Plan model has the following signal handlers:

```python
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

@receiver(post_save, sender=Plan)
def handle_plan_liked_change(sender, instance, **kwargs):
    """Create or delete embeddings when liked status changes."""
    if instance.liked:
        # Create embedding for similarity search
        create_plan_embedding(instance)
    else:
        # Delete embedding
        delete_plan_embedding(instance)
```

### 5. Unit Tests

Create tests in `/backend/tests/plans/test_views.py`:

```python
class TestPlanUpvoteDownvote(TestCase):
    def test_upvote_sets_liked_true(self):
        """Test that upvote endpoint sets liked=True."""
        plan = Plan.objects.create(...)
        response = self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        self.assertEqual(response.status_code, 200)
        plan.refresh_from_db()
        self.assertTrue(plan.liked)

    def test_downvote_sets_liked_false(self):
        """Test that downvote endpoint sets liked=False."""
        plan = Plan.objects.create(liked=True, ...)
        response = self.client.post(f'/api/v0/plans/{plan.id}/downvote/')
        self.assertEqual(response.status_code, 200)
        plan.refresh_from_db()
        self.assertFalse(plan.liked)

    def test_upvote_creates_embedding(self):
        """Test that upvoting creates an embedding."""
        plan = Plan.objects.create(...)
        self.assertEqual(PlanEmbedding.objects.filter(plan=plan).count(), 0)
        self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        self.assertEqual(PlanEmbedding.objects.filter(plan=plan).count(), 1)

    def test_downvote_deletes_embedding(self):
        """Test that downvoting deletes the embedding."""
        plan = Plan.objects.create(liked=True, ...)
        # Assume embedding was created by signal
        self.assertEqual(PlanEmbedding.objects.filter(plan=plan).count(), 1)
        self.client.post(f'/api/v0/plans/{plan.id}/downvote/')
        self.assertEqual(PlanEmbedding.objects.filter(plan=plan).count(), 0)

    def test_permissions_org_only(self):
        """Test that only org members can upvote/downvote."""
        plan = Plan.objects.create(...)
        # Test without authentication
        response = self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        self.assertEqual(response.status_code, 403)

    def test_nonexistent_plan_returns_404(self):
        """Test that upvoting non-existent plan returns 404."""
        response = self.client.post('/api/v0/plans/invalid-id/upvote/')
        self.assertEqual(response.status_code, 404)

    def test_idempotency(self):
        """Test that upvoting an already upvoted plan is idempotent."""
        plan = Plan.objects.create(...)
        self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        plan.refresh_from_db()
        self.assertTrue(plan.liked)
        # Should still have only one embedding
        self.assertEqual(PlanEmbedding.objects.filter(plan=plan).count(), 1)
```

### 6. Integration Tests

Test the full flow from API call to embedding creation/deletion:

```python
class TestPlanUpvoteIntegration(TestCase):
    def test_upvote_embedding_lifecycle(self):
        """Test complete upvote to embedding creation flow."""
        plan = Plan.objects.create(...)

        # Upvote the plan
        response = self.client.post(f'/api/v0/plans/{plan.id}/upvote/')
        self.assertEqual(response.status_code, 200)

        # Verify embedding was created
        embedding = PlanEmbedding.objects.get(plan=plan)
        self.assertIsNotNone(embedding.vector)

        # Verify plan can be found via similarity search
        similar_plans = search_similar_plans(plan.query)
        self.assertIn(plan, similar_plans)
```

## Important Considerations

### 1. Storage Eligibility
- According to the PRD, only cloud-stored plans (Storage.CLOUD) are eligible for upvotes
- Consider adding a check in the endpoint or at the model level

### 2. Backwards Compatibility
- Keep the existing PATCH endpoint for metadata updates
- The new endpoints provide better semantics but don't replace the generic PATCH

### 3. Response Format
Consider returning additional metadata:
```python
return Response({
    'status': 'success',
    'plan_id': str(plan.id),
    'liked': plan.liked,
    'embedding_created': True,  # or False
}, status=status.HTTP_200_OK)
```

### 4. Possible Enhancements
- Add a `toggle_like` endpoint that switches between upvote/downvote
- Return total count of upvoted plans in the response
- Add rate limiting for upvote/downvote operations
- Consider adding webhooks for embedding creation events

## SDK Usage Examples

Once the backend endpoints are implemented, SDK users can use the following methods:

```python
from portia import Portia
from portia.config import Config, StorageClass

# Initialize Portia with cloud storage
config = Config(
    portia_api_key="your-api-key",
    storage_class=StorageClass.CLOUD
)
portia = Portia(config=config)

# Upvote a plan
plan_id = "plan-12345678-1234-5678-1234-567812345678"
portia.upvote_plan(plan_id)

# Downvote a plan
portia.downvote_plan(plan_id)

# Async versions are also available
await portia.aupvote_plan(plan_id)
await portia.adownvote_plan(plan_id)
```

## Verification

After implementing the backend endpoints, verify they work by:

1. Running the backend unit tests
2. Running the SDK tests (they will call the actual API endpoints)
3. Testing manually via Swagger UI at `/api/docs/`
4. Checking that embeddings are created/deleted as expected
5. Verifying similarity search works with upvoted plans

## Files Changed in This PR (SDK Side)

1. `portia/storage.py` - Added `upvote_plan`, `downvote_plan`, `aupvote_plan`, `adownvote_plan` methods to `PortiaCloudStorage`
2. `portia/portia.py` - Added convenience methods in the main `Portia` class
3. `tests/unit/test_storage.py` - Added comprehensive tests for storage methods
4. `tests/unit/test_portia.py` - Added tests for Portia class convenience methods

## Next Steps

1. Implement the backend API endpoints as described above
2. Run backend tests to ensure endpoints work correctly
3. Run SDK tests to verify integration
4. Update API documentation in Swagger
5. Deploy to staging for testing
6. Deploy to production

## Questions?

If you have questions about the SDK implementation or need clarification on any requirements, please reach out to the SDK team.
