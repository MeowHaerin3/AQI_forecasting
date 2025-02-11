CREATE TABLE if NOT EXISTS users (
    id serial PRIMARY KEY,
    username VARCHAR NOT NULL,
    email VARCHAR(255) NOT NULL
);

INSERT INTO users (username, email) VALUES
    ('tanawat_s' , 'tanawat14795@gmail.com');

INSERT INTO users (id, username, email);

VALUES (
    ('32',
    'username:jeje',
    'email:jeje123@kmitl.ac.th'
    )
  );