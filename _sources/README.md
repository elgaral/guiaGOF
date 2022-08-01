# guiaGOFonline

Repositorio de los Jupyter Notebooks desarrollados en el grupo para resolver problemas puntuales que se presentan en el laboratorio

El repositorio es un libro editado en Jupyter-book.



# Guía para actualizar el Jupyter-notebook en la Internet

Luego de realizar las actualizaciones en los archivos se procede de la siguiente manera:

1. Usando un Powershell Prompt se ingresa al ambiente (enviroment) del libro: `conda activate Libro`

2. Luego se entra a la carpeta que contiene la carpeta del libro y se actualiza el jb: `jupyter-book build guiaGofOnline/`

3. Ahora se actulizan los archivos en Github entrando a la carpeta del libro: `git add ./*` `git commit -m 'comentario'``git push`

4. Finalmente se actualiza la página web: `ghp-import -n -p -f _build/html`
