from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFMinerPDFasHTMLLoader, PDFPlumberLoader

loader_PDFMinerPDFasHTMLLoader = PDFMinerPDFasHTMLLoader("~/Downloads/2401.10040.pdf")
loader_PDFPlumberLoader = PDFPlumberLoader("~/Downloads/2401.10040.pdf")
loader_PyMuPDFLoader = PyMuPDFLoader("~/Downloads/2401.10040.pdf")
loader_PyPDFLoader = PyPDFLoader("~/Downloads/2401.10040.pdf")

data_PDFMinerPDFasHTMLLoader = loader_PDFMinerPDFasHTMLLoader.load()[0]
data_PDFPlumberLoader = loader_PDFPlumberLoader.load()
data_PyMuPDFLoader = loader_PyMuPDFLoader.load()
data_PyPDFLoader = loader_PyPDFLoader.load()

import pdb; pdb.set_trace()