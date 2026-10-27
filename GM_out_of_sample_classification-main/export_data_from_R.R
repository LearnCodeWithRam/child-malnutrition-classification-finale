####### Export gorilla skull data from R to Python
####### Run this script in R to create data files for Python

# Install required packages if needed
if (!require("shapes")) install.packages("shapes")
if (!require("reticulate")) install.packages("reticulate")

library(shapes)
library(reticulate)

cat("Exporting gorilla skull data for Python...\n\n")

# Method 1: Export as NumPy arrays (recommended)
cat("Method 1: Exporting as NumPy arrays (.npy files)...\n")
tryCatch({
  np <- import("numpy")
  np$save("gorf_data.npy", gorf.dat)
  np$save("gorm_data.npy", gorm.dat)
  cat(" ✓ Created gorf_data.npy\n")
  cat(" ✓ Created gorm_data.npy\n")
}, error = function(e) {
  cat(" ✗ Failed to export NumPy arrays\n")
  cat(" Error:", conditionMessage(e), "\n")
  cat(" Make sure Python and NumPy are installed\n")
})

cat("\n")

# Method 2: Export as CSV (alternative)
cat("Method 2: Exporting as CSV files (alternative)...\n")

# Function to convert 3D array to data frame
array_to_df <- function(arr, name) {
  k <- dim(arr)[1]
  n <- dim(arr)[3]
  df <- data.frame()
  for (i in 1:n) {
    temp <- data.frame(
      individual = i,
      landmark = 1:k,
      x = arr[, 1, i],
      y = arr[, 2, i]
    )
    df <- rbind(df, temp)
  }
  return(df)
}

tryCatch({
  write.csv(array_to_df(gorf.dat, "female"), "gorf_data.csv", row.names = FALSE)
  write.csv(array_to_df(gorm.dat, "male"), "gorm_data.csv", row.names = FALSE)
  cat("  ✓ Created gorf_data.csv\n")
  cat("  ✓ Created gorm_data.csv\n")
}, error = function(e) {
  cat("  ✗ Failed to export CSV files\n")
  cat("  Error:", conditionMessage(e), "\n")
})

cat("\n")

# Display data dimensions
cat("Data dimensions:\n")
cat("  Female gorillas (gorf.dat):\n")
cat("    Landmarks:", dim(gorf.dat)[1], "\n")
cat("    Dimensions:", dim(gorf.dat)[2], "\n")
cat("    Individuals:", dim(gorf.dat)[3], "\n")
cat("\n")
cat("  Male gorillas (gorm.dat):\n")
cat("    Landmarks:", dim(gorm.dat)[1], "\n")
cat("    Dimensions:", dim(gorm.dat)[2], "\n")
cat("    Individuals:", dim(gorm.dat)[3], "\n")

cat("\n")
cat("============================================================\n")
cat("Export complete!\n")
cat("You can now run the Python code with this data.\n")
cat("============================================================\n")
