setwd("~/Documents/optimalcausalities-software")

package_path <- "~/Documents/optimalcausalities-software/optimalcausalities"

devtools::document(package_path)
try(file.remove(sprintf("./optimalcausalities.pdf")),T)
system(paste(shQuote(file.path(R.home("bin"), "R")),
             "CMD", "Rd2pdf", shQuote(package_path)))

#install.packages(package_path)

#install.packages( "~/Documents/optimalcausalities-software/optimalcausalities",repos = NULL, type = "source")
