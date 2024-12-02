package com.example.sidewalktest

import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.sidewalktest.ml.Sidewalkmodel
import com.example.sidewalktest.ml.Regressionmodel
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var sidewalkModel: Sidewalkmodel
    private lateinit var regressionModel: Regressionmodel
    private val SELECT_IMAGE_REQUEST = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        val selectImageButton: Button = findViewById(R.id.selectImageButton)

        // Initialize TensorFlow Lite models
        sidewalkModel = Sidewalkmodel.newInstance(this)
        regressionModel = Regressionmodel.newInstance(this)

        selectImageButton.setOnClickListener {
            openImagePicker()
        }
    }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        intent.type = "image/*"
        startActivityForResult(intent, SELECT_IMAGE_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SELECT_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            val imageUri: Uri? = data.data
            if (imageUri != null) {
                processSelectedImage(imageUri)
            }
        }
    }

    private fun processSelectedImage(imageUri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(imageUri)
            val originalBitmap = BitmapFactory.decodeStream(inputStream)

            // Resize the bitmap to avoid memory issues
            val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 1024, 1024, true)

            // Display the resized image
            imageView.setImageBitmap(resizedBitmap)

            // Run the models in parallel
            runModelsInParallel(resizedBitmap)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun runModelsInParallel(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build()

                val tensorImage = TensorImage.fromBitmap(bitmap)
                val processedImage = imageProcessor.process(tensorImage)

                // Run models concurrently
                val sidewalkDeferred = async { runSidewalkModel(processedImage, bitmap) }
                val regressionDeferred = async { runRegressionModel(processedImage) }

                val sidewalkBitmap = sidewalkDeferred.await()
                val severity = regressionDeferred.await()

                // Update UI
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(sidewalkBitmap)
                    displaySeverityPrompt(severity)
                }
            } catch (e: Exception) {
                Log.e("ModelError", "Error running models", e)
            }
        }
    }

    private suspend fun runSidewalkModel(processedImage: TensorImage, bitmap: Bitmap): Bitmap {
        return withContext(Dispatchers.IO) {
            val sidewalkOutputs = sidewalkModel.process(processedImage)
            val sidewalkOutputArray = sidewalkOutputs.outputAsTensorBuffer.floatArray

            val width = bitmap.width.toFloat()
            val height = bitmap.height.toFloat()

            drawSidewalkDetections(bitmap, sidewalkOutputArray, width, height)
        }
    }

    private fun drawSidewalkDetections(bitmap: Bitmap, outputArray: FloatArray, width: Float, height: Float): Bitmap {
        val stride = 6
        val threshold = 1.0f // Confidence threshold
        val iouThreshold = 0.1f // Intersection over Union (IoU) threshold for NMS

        val detections = mutableListOf<Detection>()

        // Parse detections from output array
        for (i in outputArray.indices step stride) {
            val xMin = (outputArray[i] * width).toInt()
            val yMin = (outputArray[i + 1] * height).toInt()
            val xMax = (outputArray[i + 2] * width).toInt()
            val yMax = (outputArray[i + 3] * height).toInt()
            val confidence = outputArray[i + 4]

            if (confidence >= threshold) {
                detections.add(Detection(xMin, yMin, xMax, yMax, confidence))
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        val filteredDetections = nonMaximumSuppression(detections, iouThreshold)

        // Draw filtered detections
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()

        for (detection in filteredDetections) {
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 5f
            canvas.drawRect(RectF(detection.xMin.toFloat(), detection.yMin.toFloat(), detection.xMax.toFloat(), detection.yMax.toFloat()), paint)

            paint.style = Paint.Style.FILL
            paint.textSize = 40f
            canvas.drawText("Confidence: ${"%.2f".format(detection.confidence)}", detection.xMin.toFloat(), (detection.yMin - 10f), paint)
        }

        return mutableBitmap
    }

    // Data class for detection
    data class Detection(
        val xMin: Int,
        val yMin: Int,
        val xMax: Int,
        val yMax: Int,
        val confidence: Float
    )

    // Non-Maximum Suppression (NMS) implementation
    private fun nonMaximumSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val filteredDetections = mutableListOf<Detection>()
        val sortedDetections = detections.sortedByDescending { it.confidence }.toMutableList() // Convert to MutableList

        while (sortedDetections.isNotEmpty()) {
            val bestDetection = sortedDetections.first()
            filteredDetections.add(bestDetection)
            sortedDetections.removeAt(0) // Remove the first (highest confidence) detection

            sortedDetections.removeAll { detection ->
                calculateIoU(bestDetection, detection) > iouThreshold
            }
        }

        return filteredDetections
    }

    // Calculate Intersection over Union (IoU)
    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val intersectionXMin = max(det1.xMin, det2.xMin)
        val intersectionYMin = max(det1.yMin, det2.yMin)
        val intersectionXMax = min(det1.xMax, det2.xMax)
        val intersectionYMax = min(det1.yMax, det2.yMax)

        val intersectionArea = max(0, intersectionXMax - intersectionXMin) * max(0, intersectionYMax - intersectionYMin)
        val areaDet1 = (det1.xMax - det1.xMin) * (det1.yMax - det1.yMin)
        val areaDet2 = (det2.xMax - det2.xMin) * (det2.yMax - det2.yMin)

        val unionArea = areaDet1 + areaDet2 - intersectionArea
        return if (unionArea > 0) intersectionArea.toFloat() / unionArea else 0f
    }

    private suspend fun runRegressionModel(processedImage: TensorImage): Float {
        return withContext(Dispatchers.IO) {
            try {
                // Ensure the image is resized and normalized correctly
                val regressionProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .build()

                val regressionInput = regressionProcessor.process(processedImage)

                // Create TensorBuffer for input
                val inputTensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                inputTensorBuffer.loadBuffer(regressionInput.tensorBuffer.buffer)

                // Run the Regression Model
                val regressionOutputs = regressionModel.process(inputTensorBuffer)
                regressionOutputs.outputFeature0AsTensorBuffer.floatArray[0]
            } catch (e: Exception) {
                Log.e("RegressionModelError", "Error running regression model", e)
                -1f // Return -1 for errors
            }
        }
    }

    private fun displaySeverityPrompt(severity: Float) {
        val severityTextView = findViewById<TextView>(R.id.severityTextView)

        val severityText = when {
            severity <= 1 -> "Low Severity. Stay cautious."
            severity <= 2 -> "Moderate Severity. Stay alert."
            severity <= 3 -> "High Severity. Be very cautious."
            severity <= 4 -> "Critical Severity. Immediate action required!"
            else -> "Error calculating severity."
        }

        Log.d("SeverityPrompt", severityText)
        severityTextView.text = severityText
    }

    override fun onDestroy() {
        super.onDestroy()
        sidewalkModel.close()
        regressionModel.close()
    }
}