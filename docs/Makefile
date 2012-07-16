WSOM_Paper.dvi: WSOM_Paper.tex
	pdflatex WSOM_Paper.tex
	bibtex WSOM_Paper.aux
	pdflatex WSOM_Paper.tex
	pdflatex WSOM_Paper.tex

view: WSOM_Paper.dvi
	okular WSOM_Paper.pdf

clean:
	rm *.aux
	rm *.bbl
	rm *.blg
	rm *.dvi
	rm *.log
