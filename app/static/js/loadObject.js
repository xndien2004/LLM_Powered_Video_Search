fetch('/api/objects/', {
    method: 'GET',
})
.then(response => response.json())
.then(data => {
    if (data.error) {
        alert(data.error);
    } else {
        console.log(data); 
        image_objects = data.image_objects;

        const container = document.querySelector('.icon-container');
        image_objects.forEach(icon => {
            const iconBox = document.createElement('div');
            iconBox.className = 'icon-box';
            iconBox.dataset.imgSrc = icon.src;
    
            const img = document.createElement('img');
            img.src = icon.src;
            img.title = icon.name;
            img.id = icon.name.toLowerCase(); 
    
            const nameDiv = document.createElement('div');
            nameDiv.className = 'name-icon';
            nameDiv.textContent = icon.name;
    
            iconBox.appendChild(img);
            iconBox.appendChild(nameDiv);
    
            container.appendChild(iconBox);
        });
    }
})
.catch(console.log);

fetch('/api/colors/', {
    method: 'GET',
})
.then(response => response.json())
.then(data => {
    if (data.error) {
        alert(data.error);
    } else {
        console.log(data); 
        image_objects = data.image_objects;

        const container = document.querySelector('.color-container');
        image_objects.forEach(color => {
            const colorBox = document.createElement('div');
            colorBox.className = 'color-box';
            colorBox.dataset.imgSrc = color.src;
    
            const img = document.createElement('img');
            img.src = color.src;
            img.title = color.name;
            img.id = color.name.toLowerCase();  
    
            const nameDiv = document.createElement('div');
            nameDiv.className = 'name-color';
            nameDiv.textContent = color.name;
    
            colorBox.appendChild(img);
            colorBox.appendChild(nameDiv);
    
            container.appendChild(colorBox);
        });
        var colorNames = [];
        image_objects.forEach(color => {
            colorNames.push(color.name);
        });
        localStorage.setItem('colorNames', JSON.stringify(colorNames));
    }
})
.catch(console.log);

// load tag
fetch('/api/tag_query/', {
    method: 'GET',
})
.then(response => response.json())
.then(data => {
    if (data.error) {
        alert(data.error);
    } else {
        console.log(data); 
        let tags = data.tags;

        const tagBox = document.querySelector('.tag-container .tag-box');
        tags.forEach(tag => {
            const tagDiv = document.createElement('div');
            tagDiv.className = 'tag';
            tagDiv.textContent = tag;
            tagDiv.dataset.tag = tag;
            tagBox.appendChild(tagDiv);
        });
    }
})
.catch(console.log);