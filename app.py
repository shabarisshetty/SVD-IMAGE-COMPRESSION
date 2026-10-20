import streamlit as st
import numpy as np

st.title("Advanced SVD Image Compression Analyzer")
st.markdown("""
This application demonstrates how Singular Value Decomposition (SVD)
can be used for image compression.

The image is converted into a grayscale matrix, decomposed using SVD,
and reconstructed using a selected number of singular values (k).

Smaller k → higher compression but larger reconstruction error.

Larger k → better image quality but lower compression.
""")

st.write("Upload an image to begin SVD-based compression analysis.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.success("Image uploaded successfully!")

    from PIL import Image

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    gray_image = image.convert("L")

    st.image(
        gray_image,
        caption="Grayscale Image",
        use_container_width=True
    )
    image_matrix = np.array(gray_image)
    

st.subheader("Image Matrix")

st.write("Matrix dimensions:", image_matrix.shape)

st.write("Sample pixel values:")

st.write(image_matrix[:10, :10])
matrix_rank = np.linalg.matrix_rank(image_matrix)

st.subheader("Matrix Rank")

st.write("Rank of the image matrix:", matrix_rank)
U, S, VT = np.linalg.svd(
    image_matrix,
    full_matrices=False
)

st.subheader("SVD Analysis")

st.write("U matrix shape:", U.shape)
st.write("Number of singular values:", len(S))
st.write("Vᵀ matrix shape:", VT.shape)
st.subheader("Singular Value Distribution")

st.line_chart(S)
st.subheader("Choose Compression Level")

k = st.slider(
    "Number of singular values (k)",
    min_value=1,
    max_value=len(S),
    value=min(50, len(S))
)

st.write("Selected k:", k)
compressed_matrix = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

compressed_matrix = np.clip(
    compressed_matrix,
    0,
    255
).astype(np.uint8)

compressed_image = Image.fromarray(compressed_matrix)

st.subheader("Compressed / Reconstructed Image")

st.image(
    compressed_image,
    caption=f"Reconstructed Image (k = {k})",
    use_container_width=True
)
error = np.linalg.norm(image_matrix - compressed_matrix) / np.linalg.norm(image_matrix)

st.subheader("Reconstruction Error")

st.write(f"Relative Reconstruction Error: {error:.6f}")
st.subheader("Comparison of Different Compression Levels")

comparison_values = [10, 25, 50, 100]

for value in comparison_values:
    if value <= len(S):
        comparison_matrix = U[:, :value] @ np.diag(S[:value]) @ VT[:value, :]
        comparison_matrix = np.clip(comparison_matrix, 0, 255).astype(np.uint8)

        comparison_error = (
            np.linalg.norm(image_matrix - comparison_matrix)
            / np.linalg.norm(image_matrix)
        )

        st.write(f"### k = {value}")
        st.image(
            comparison_matrix,
            caption=f"Reconstructed Image — k = {value}",
            use_container_width=True
        )
        st.write(f"Reconstruction Error: {comparison_error:.6f}")
        st.subheader("Reconstruction Error Comparison")

k_values = [10, 25, 50, 100]
errors = []

for value in k_values:
    if value <= len(S):
        temp_matrix = U[:, :value] @ np.diag(S[:value]) @ VT[:value, :]

        temp_error = (
            np.linalg.norm(image_matrix - temp_matrix)
            / np.linalg.norm(image_matrix)
        )

        errors.append(temp_error)

st.line_chart(
    {
        "k": k_values[:len(errors)],
        "Reconstruction Error": errors
    }
)
st.subheader("Compression / Storage Analysis")

rows, cols = image_matrix.shape

original_storage = rows * cols

storage_values = []
storage_reduction = []

for value in k_values:
    if value <= len(S):
        compressed_storage = value * (rows + cols + 1)

        storage_values.append(compressed_storage)

        reduction = (1 - compressed_storage / original_storage) * 100
        storage_reduction.append(reduction)

st.write(f"Original image matrix size: {rows} × {cols}")
st.write(f"Original number of values: {original_storage:,}")

for i, value in enumerate(k_values):
    if value <= len(S):
        st.write(
            f"k = {value}: "
            f"Approx. storage = {storage_values[i]:,} values, "
            f"Storage reduction = {storage_reduction[i]:.2f}%"
        )
        st.subheader("Project Information")

st.write("""
**Method Used:** Singular Value Decomposition (SVD)

**Input:** Grayscale image represented as a matrix

**Main Concepts:**
- Matrix Rank
- Singular Values
- SVD Decomposition
- Low-Rank Approximation
- Image Reconstruction
- Reconstruction Error

**Purpose:** To analyze the trade-off between image quality and
compression by varying the number of singular values (k).
""")