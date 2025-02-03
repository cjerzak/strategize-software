{
  setwd("~/Documents/strategize-software"); options(error = NULL)
  
  package_path <- "~/Documents/strategize-software/strategize"
  
  devtools::document(package_path)
  try(file.remove(sprintf("./strategize.pdf")),T)
  system(paste(shQuote(file.path(R.home("bin"), "R")),
               "CMD", "Rd2pdf", shQuote(package_path)))
  
  # Check package to ensure it meets CRAN standards.
  # devtools::check( package_path )

  # install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
}
