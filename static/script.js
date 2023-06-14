const genreButtons = document.querySelectorAll(".button");
const resultsDiv = document.getElementById("results");

genreButtons.forEach((button) => {
    button.addEventListener("click", () => {
        button.classList.toggle("button-selected");
        recommendAnimes();
    });
});

function recommendAnimes() {
    const selectedGenres = Array.from(genreButtons)
        .filter((button) => button.classList.contains("button-selected"))
        .map((button) => button.dataset.genre);

    fetch("/recommend", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ user_genres: selectedGenres.join(", ") }),
    })
        .then((response) => response.json())
        .then((data) => {
            const recommendedAnimes = data.results;
            const cards = recommendedAnimes.map((anime) => createAnimeCard(anime)).join("");
            resultsDiv.innerHTML = cards;
        });
}

function createAnimeCard(anime) {
    return `
        <div class="anime-card">
            <img src="${anime.img_url}" alt="${anime.title}" class="anime-image">
            <div class="anime-title">${anime.title}</div>
            <div class="anime-genre">${anime.genre.join(", ")}</div>
            <div class="anime-synopsis">${anime.synopsis}</div>
            <a href="${anime.link}" target="_blank" class="anime-link">View on MyAnimeList</a>
        </div>
    `;
}