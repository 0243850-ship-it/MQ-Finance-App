import streamlit as st
from pathlib import Path

# ---------- Config ----------
st.set_page_config(page_title="CV - Olga Mariana Quezada Sánchez", page_icon="📄", layout="centered")

# ---------- Datos del CV (solo lo que está en tu curriculum) ----------
NOMBRE = "OLGA MARIANA QUEZADA SÁNCHEZ"
TITULO = "Administración y Finanzas"

CONTACTO = [
    ("📍 Ubicación", "Zapopan, Jalisco"),
    ("📞 Teléfono", "(311) 147 10 08"),
    ("✉️ Email", "marianaqs0519@gmail.com"),
]

EDUCACION = [
    ("Universidad Panamericana", "Licenciatura en Administración y Finanzas", "2021 - Actualidad"),
    ("Colegio Cristóbal Colón", "Primaria, Secundaria y Preparatoria", "2009 - 2021"),
]

EXPERIENCIA = [
    ("Laboratorios Clínicos Quezada", "Auxiliar en contabilidad", "Junio 2023 - Agosto 2023"),
    ("Centro de Expresión Artística iDance", "Auxiliar de almacén", "Junio 2019 - Agosto 2019"),
    ("Laboratorios Clínicos Quezada", "Maestra de baile moderno y contemporáneo", "Julio 2018 - Diciembre 2020"),
]

IDIOMAS = [
    "Español — Nativo",
    "Inglés — Avanzado",
]

HABILIDADES = [
    "Word", "PowerPoint", "Excel", "Danza Artística", "Debate",
]

ACTIVIDADES = [
    "Curso intensivo de Inglés en Toronto, Ontario, Canadá (Centre of English Studies, 4 - 29 Julio 2022).",
    "Certificación de Inglés ESOL (Cambridge Assessment English, Junio 2018).",
    "Curso de Inglés (Planet English, Agosto 2016 - Noviembre 2017).",
]

# ---------- Encabezado ----------
st.title(NOMBRE)
st.caption(TITULO)

# Foto (colócala como foto.jpg/png/jpeg en la misma carpeta)
carpeta = Path(__file__).parent
posibles = ["foto.jpg", "foto.png", "foto.jpeg", "foto.JPG", "foto.PNG", "foto.JPEG"]
ruta_foto = next((carpeta / n for n in posibles if (carpeta / n).exists()), None)

if ruta_foto:
    st.image(str(ruta_foto), width=220)
else:
    st.info("Para mostrar tu foto, guarda una imagen en esta carpeta con el nombre **foto.jpg** (o .png/.jpeg).")

st.divider()

# ---------- Contacto ----------
st.subheader("Contacto")
for k, v in CONTACTO:
    if "Email" in k:
        st.write(f"- **{k}:** [{v}](mailto:{v})")
    else:
        st.write(f"- **{k}:** {v}")

st.divider()

# ---------- Educación ----------
st.subheader("Educación")
for inst, detalle, periodo in EDUCACION:
    st.markdown(f"**{inst}**  \n{detalle}  \n_{periodo}_")

st.divider()

# ---------- Experiencia ----------
st.subheader("Experiencia")
for lugar, puesto, periodo in EXPERIENCIA:
    st.markdown(f"**{lugar}** — {puesto}  \n_{periodo}_")

st.divider()

# ---------- Idiomas ----------
st.subheader("Idiomas")
for i in IDIOMAS:
    st.write(f"- {i}")

# ---------- Habilidades ----------
st.subheader("Habilidades")
st.write(", ".join(HABILIDADES))

# ---------- Actividades Complementarias ----------
st.subheader("Actividades complementarias")
for a in ACTIVIDADES:
    st.write(f"- {a}")

st.caption("Hecho con Streamlit")
