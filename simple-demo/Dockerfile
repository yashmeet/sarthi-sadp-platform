FROM nginx:alpine

# Copy all HTML files to nginx html directory
COPY *.html /usr/share/nginx/html/

# Copy JavaScript files
COPY *.js /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]