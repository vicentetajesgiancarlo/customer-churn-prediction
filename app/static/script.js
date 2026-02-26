document.getElementById('predictionForm').addEventListener('submit', async (evento) => {
    evento.preventDefault();

    const formulario = evento.target;
    const botonEnviando = document.getElementById('predictBtn');
    const tarjetaResultado = document.getElementById('resultCard');
    const barraDeCarga = document.getElementById('riskFill');
    const textoPorcentaje = document.getElementById('riskPercent');
    const tituloResultado = document.getElementById('resultLabel');
    const distintivoRiesgo = document.getElementById('riskBadge');
    const descripcionResultado = document.getElementById('resultDesc');

    // Estado visual de carga
    botonEnviando.disabled = true;
    botonEnviando.classList.add('loading');
    botonEnviando.innerHTML = '<span>Procesando...</span><i class="fas fa-spinner fa-spin"></i>';

    const datosFormulario = new FormData(formulario);
    const objetoDeDatos = Object.fromEntries(datosFormulario.entries());

    // Asegurar que los tipos de datos numéricos sean correctos (excluyendo nombres de columnas)
    objetoDeDatos.SeniorCitizen = parseInt(objetoDeDatos.SeniorCitizen);
    objetoDeDatos.tenure = parseInt(objetoDeDatos.tenure);
    objetoDeDatos.MonthlyCharges = parseFloat(objetoDeDatos.MonthlyCharges);
    objetoDeDatos.TotalCharges = parseFloat(objetoDeDatos.TotalCharges);

    try {
        const respuestaServidor = await fetch('http://localhost:8010/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(objetoDeDatos)
        });

        if (!respuestaServidor.ok) throw new Error('Error al conectar con la API');

        const resultadoIA = await respuestaServidor.json();
        const scoreProbabilidad = resultadoIA.churn_probability;
        const porcentajeFinal = (scoreProbabilidad * 100).toFixed(1) + '%';

        // Mostrar el contenedor de resultados
        tarjetaResultado.classList.remove('result-hidden');
        tarjetaResultado.classList.add('result-show');

        // Animación dinámica del medidor de riesgo
        setTimeout(() => {
            barraDeCarga.style.height = (scoreProbabilidad * 100) + '%';
            textoPorcentaje.innerText = porcentajeFinal;
        }, 100);

        // Actualización de textos y etiquetas basadas en el resultado
        tituloResultado.innerText = resultadoIA.prediction === 'Churn' ?
            'Predicción: EL CLIENTE CANCELARÁ' : 'Predicción: EL CLIENTE CONTINUARÁ';

        descripcionResultado.innerText = `El modelo estima un ${porcentajeFinal} de probabilidad de abandono basada en el historial ingresado.`;

        distintivoRiesgo.innerText = `Nivel de Riesgo: ${resultadoIA.risk_level}`;
        distintivoRiesgo.className = 'risk-badge'; // Resetear estilos

        if (resultadoIA.risk_level === 'High') {
            distintivoRiesgo.classList.add('risk-high');
        } else if (resultadoIA.risk_level === 'Medium') {
            distintivoRiesgo.classList.add('risk-medium');
        }

        // Desplazamiento suave hasta el resultado
        tarjetaResultado.scrollIntoView({ behavior: 'smooth', block: 'center' });

    } catch (errorCapturado) {
        console.error(errorCapturado);
        alert('Hubo un error al conectar con el servidor de inteligencia artificial.');
    } finally {
        // Restaurar estado del botón
        botonEnviando.disabled = false;
        botonEnviando.classList.remove('loading');
        botonEnviando.innerHTML = '<span>Analizar Riesgo</span><i class="fas fa-microchip"></i>';
    }
});
