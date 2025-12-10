# backend/filterx/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services.nsfw_classifier import (
    classify_image_base64,
    classify_text,
    classify_url,
    _MEDIANET_MODEL,
    MODEL_PATH,
)


class ClassifyView(APIView):
    """
    POST /api/classify/

    Request:
    {
      "type": "image" | "text" | "url",
      "content": "..."
    }
    """

    def post(self, request, *args, **kwargs):
        data = request.data
        content_type = data.get("type")
        content = data.get("content")

        if not content_type or not content:
            return Response(
                {"error": "Both 'type' and 'content' are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if content_type == "image":
            result = classify_image_base64(content)
        elif content_type == "text":
            result = classify_text(content)
        elif content_type == "url":
            result = classify_url(content)
        else:
            return Response(
                {"error": "Invalid type. Use 'image', 'text', or 'url'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(result, status=status.HTTP_200_OK)


class HealthCheckView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class StatusView(APIView):
    """Returns whether MediaNet model loaded successfully."""

    def get(self, request, *args, **kwargs):
        return Response({
            "media_net_loaded": bool(_MEDIANET_MODEL is not None)
        }, status=status.HTTP_200_OK)


class ModelFilesView(APIView):
    """List files under the model directory to help debugging where the files are."""

    def get(self, request, *args, **kwargs):
        try:
            files = []
            for path in MODEL_PATH.rglob("**/*"):
                # only show relative paths and skip directories
                if path.is_file():
                    files.append(str(path.relative_to(MODEL_PATH)))
            return Response({"model_path": str(MODEL_PATH), "files": files}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e), "model_path": str(MODEL_PATH)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
