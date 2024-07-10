
document.addEventListener('DOMContentLoaded', function () {
    const ctx = document.getElementById('myPieChart').getContext('2d');
    const data = {
        labels: [
            'Combined',
            'Impulsive / Hyperactive',
            'Inattentive',
            'No ADHD'
        ],
        datasets: [{
            label: 'Results',
            data: diagnosisDataBackend,
            backgroundColor: [
                'rgb(255, 99, 132)',
                'rgb(54, 162, 235)',
                'rgb(255, 205, 86)',
                'rgb(102, 22, 200)'
            ],
            hoverOffset: 4
        }]
    };

    const config = {
        type: 'pie',
        data: data,
    };

    const myPieChart = new Chart(ctx, config);
});
